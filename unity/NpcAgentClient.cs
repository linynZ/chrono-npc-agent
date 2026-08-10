using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace ChronoTraveler.Agent
{
    /// <summary>
    /// Talks to the chrono-npc-agent service. Deliberately free of any game
    /// references so it compiles on its own and can be lifted into another
    /// project — everything game-specific lives in <c>ChronoAgentBridge</c>.
    ///
    /// Both a blocking call and a streamed one are provided. Streaming is the
    /// one worth using: the game already renders dialogue with a typewriter, so
    /// deltas can drive it directly instead of waiting for the full reply and
    /// then animating text that has already arrived.
    ///
    /// Field names are snake_case on purpose. JsonUtility cannot rename fields,
    /// and the service speaks snake_case, so the DTOs match the wire rather than
    /// C# convention. The convention loses; a hand-written mapping layer for
    /// six structs would be worse.
    /// </summary>
    public class NpcAgentClient : MonoBehaviour
    {
        [Tooltip("Base URL of the agent service.")]
        public string baseUrl = "http://127.0.0.1:8000";

        [Tooltip("deepseek | ollama | echo")]
        public string backend = "deepseek";

        [Tooltip("Give up after this long and let the caller fall back. The service\nfalls back on its own too; this is the outer net for the service itself\nbeing unreachable.")]
        public float timeoutSeconds = 20f;

        public bool IsBusy { get; private set; }

        // --- wire types -------------------------------------------------

        [Serializable]
        public class QuestDto
        {
            public string quest_id;
            public int stage_index;
            public string stage_description;
            public string objective_label;
            public int progress;
            public int required = 1;
        }

        [Serializable]
        public class PlayerStateDto
        {
            public string current_map = "20_China";
            public string language = "zh";
            public float memory_progress;
            public int fragments_collected;
            public int fragments_required = 3;
            public string[] purged_encounters = Array.Empty<string>();
            public string[] talked_npcs = Array.Empty<string>();
            public string[] earned_tokens = Array.Empty<string>();
            public int player_level = 1;
            public int correct_answers;
            public int total_answers;
            public QuestDto quest;
        }

        [Serializable]
        public class TurnDto
        {
            public string role;
            public string text;
            public TurnDto() { }
            public TurnDto(string role, string text) { this.role = role; this.text = text; }
        }

        [Serializable]
        private class ChatRequestDto
        {
            public string message;
            public string npc_id;
            public PlayerStateDto state;
            public TurnDto[] history;
            public string language;
            public string backend;
            public bool free_form;
        }

        [Serializable]
        public class ReplyDto
        {
            public string text;
            public string speaker;
            public string source;          // model | fallback_timeout | fallback_error | fallback_guardrail
            public float latency_ms;
            public float first_token_ms;
            public string[] tool_calls;
            public string[] guardrail_flags;
            public int tokens;
            public string backend;
            public string model;

            public bool IsFallback => !string.IsNullOrEmpty(source) && source.StartsWith("fallback");
        }

        [Serializable]
        private class StreamFrame
        {
            public string kind;            // delta | replace | done | error
            public string text;
            public ReplyDto reply;
        }

        [Serializable]
        public class NpcInfoDto
        {
            public string npc_id;
            public string name;
            public string era;
            public string[] openers;
            public string[] tools;
        }

        // --- discovery --------------------------------------------------

        /// <summary>
        /// Ask the service what it knows about an NPC. Used to decide whether to
        /// offer free conversation at all: a failure here means the affordance is
        /// simply not shown, which is the right outcome — an entry point that
        /// does nothing when pressed is worse than no entry point.
        /// </summary>
        public IEnumerator FetchNpcInfo(
            string npcId, string language,
            Action<NpcInfoDto> onDone, Action<string> onError = null)
        {
            string url = $"{baseUrl.TrimEnd('/')}/api/npc/{npcId}?language={language}";
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = Mathf.Max(1, Mathf.CeilToInt(timeoutSeconds));
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(Describe(request));
                    yield break;
                }

                NpcInfoDto info = null;
                try { info = JsonUtility.FromJson<NpcInfoDto>(request.downloadHandler.text); }
                catch (Exception e) { onError?.Invoke("bad response: " + e.Message); }

                if (info != null) onDone?.Invoke(info);
            }
        }

        // --- blocking ---------------------------------------------------

        /// <summary>
        /// One turn, delivered whole. Simpler, and ~1.5x longer before the
        /// player sees anything. Use <see cref="ChatStream"/> unless the caller
        /// genuinely cannot render incrementally.
        /// </summary>
        public IEnumerator Chat(
            string npcId, string message, PlayerStateDto state,
            IList<TurnDto> history, Action<ReplyDto> onDone, Action<string> onError = null,
            bool freeForm = false)
        {
            IsBusy = true;
            using (UnityWebRequest request = BuildRequest("/api/chat", npcId, message, state, history, freeForm))
            {
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    IsBusy = false;
                    onError?.Invoke(Describe(request));
                    yield break;
                }

                ReplyDto reply = null;
                try { reply = JsonUtility.FromJson<ReplyDto>(request.downloadHandler.text); }
                catch (Exception e) { onError?.Invoke("bad response: " + e.Message); }

                IsBusy = false;
                if (reply != null) onDone?.Invoke(reply);
            }
        }

        // --- streaming --------------------------------------------------

        /// <summary>
        /// One turn, delivered as it is produced.
        ///
        /// <paramref name="onReplace"/> is not optional. It is how the service
        /// retracts text already on screen when a guardrail trips mid-reply —
        /// ignoring it would leave a leaked quiz answer in front of the player.
        /// Whatever the callback receives becomes the whole visible line,
        /// replacing everything shown so far (an empty string clears it).
        /// </summary>
        public IEnumerator ChatStream(
            string npcId, string message, PlayerStateDto state, IList<TurnDto> history,
            Action<string> onDelta, Action<string> onReplace, Action<ReplyDto> onDone,
            Action<string> onError = null, bool freeForm = false)
        {
            IsBusy = true;

            ReplyDto finished = null;
            string streamError = null;

            void HandleFrame(string json)
            {
                StreamFrame frame;
                try { frame = JsonUtility.FromJson<StreamFrame>(json); }
                catch { return; }
                if (frame == null || string.IsNullOrEmpty(frame.kind)) return;

                switch (frame.kind)
                {
                    case "delta":   onDelta?.Invoke(frame.text ?? string.Empty); break;
                    case "replace": onReplace?.Invoke(frame.text ?? string.Empty); break;
                    case "done":    finished = frame.reply; break;
                    case "error":   streamError = frame.text; break;
                }
            }

            using (UnityWebRequest request = BuildRequest("/api/chat/stream", npcId, message, state, history, freeForm))
            {
                request.downloadHandler = new SseHandler(HandleFrame);
                yield return request.SendWebRequest();

                IsBusy = false;

                if (request.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(Describe(request));
                    yield break;
                }
                if (streamError != null) { onError?.Invoke(streamError); yield break; }
                if (finished == null) { onError?.Invoke("stream ended without a result"); yield break; }

                onDone?.Invoke(finished);
            }
        }

        // --- plumbing ---------------------------------------------------

        private UnityWebRequest BuildRequest(
            string path, string npcId, string message,
            PlayerStateDto state, IList<TurnDto> history, bool freeForm = false)
        {
            var payload = new ChatRequestDto
            {
                message = message,
                npc_id = npcId,
                state = state ?? new PlayerStateDto(),
                history = ToArray(history),
                language = state != null ? state.language : "zh",
                backend = backend,
                // Free conversation degrades to the persona's last-resort line
                // rather than replaying an unrelated scripted beat.
                free_form = freeForm
            };

            var request = new UnityWebRequest(baseUrl.TrimEnd('/') + path, UnityWebRequest.kHttpVerbPOST);
            byte[] body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload));
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = Mathf.Max(1, Mathf.CeilToInt(timeoutSeconds));
            return request;
        }

        private static TurnDto[] ToArray(IList<TurnDto> history)
        {
            if (history == null || history.Count == 0) return Array.Empty<TurnDto>();
            var copy = new TurnDto[history.Count];
            for (int i = 0; i < history.Count; i++) copy[i] = history[i];
            return copy;
        }

        private static string Describe(UnityWebRequest request)
        {
            return request.responseCode > 0
                ? $"HTTP {request.responseCode}: {request.error}"
                : $"{request.error} (is the agent service running?)";
        }

        /// <summary>
        /// Reassembles server-sent events out of whatever chunk sizes the
        /// transport hands over.
        ///
        /// The decoder is stateful for a reason: a UTF-8 Chinese character is
        /// three bytes and lands across a chunk boundary often enough to matter.
        /// Calling Encoding.UTF8.GetString per chunk would corrupt exactly those
        /// characters, which in this game is most of them.
        /// </summary>
        private sealed class SseHandler : DownloadHandlerScript
        {
            private readonly Action<string> onFrame;
            private readonly Decoder decoder = Encoding.UTF8.GetDecoder();
            private readonly StringBuilder buffer = new StringBuilder();
            private char[] chars = new char[2048];

            public SseHandler(Action<string> onFrame) : base(new byte[4096])
            {
                this.onFrame = onFrame;
            }

            protected override bool ReceiveData(byte[] data, int dataLength)
            {
                if (data == null || dataLength == 0) return false;

                int needed = decoder.GetCharCount(data, 0, dataLength, false);
                if (needed > chars.Length) chars = new char[needed];
                int written = decoder.GetChars(data, 0, dataLength, chars, 0, false);
                buffer.Append(chars, 0, written);

                Drain();
                return true;
            }

            private void Drain()
            {
                while (true)
                {
                    string pending = buffer.ToString();
                    int split = pending.IndexOf("\n\n", StringComparison.Ordinal);
                    if (split < 0) return;

                    string frame = pending.Substring(0, split);
                    buffer.Remove(0, split + 2);

                    frame = frame.Trim();
                    if (frame.StartsWith("data:", StringComparison.Ordinal))
                        onFrame(frame.Substring(5).Trim());
                }
            }
        }
    }
}

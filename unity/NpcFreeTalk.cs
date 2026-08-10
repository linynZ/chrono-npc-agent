using System.Collections;
using System.Collections.Generic;
using ChronoTraveler.Battle;
using ChronoTraveler.Core;
using ChronoTraveler.Dialogue;
using ChronoTraveler.UI;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ChronoTraveler.Agent
{
    /// <summary>
    /// Free conversation, layered on top of the written dialogue rather than
    /// replacing it.
    ///
    /// The scripted lines still run first and still fire the quest events that
    /// depend on them — <c>OnDialogueCompletedId</c> is what advances a TalkTo
    /// objective, so an NPC whose dialogue was handed wholesale to a model would
    /// quietly break the quest chain. When the written lines finish, this offers
    /// to keep talking. Everything the player already had is untouched; this is
    /// only ever an addition.
    ///
    /// The entry point appears only when the service answered <c>/api/npc/{id}</c>
    /// for this NPC. A prompt that does nothing when pressed is worse than no
    /// prompt, so with the service down the game behaves exactly as it always did.
    ///
    /// Self-booting and self-building, like the game's other UI singletons —
    /// nothing to attach, nothing to wire in a scene.
    /// </summary>
    public class NpcFreeTalk : MonoBehaviour
    {
        public const KeyCode OpenKey = KeyCode.T;
        private const int SortOrder = 260;      // just above UILayerOrder.Dialogue (250)
        private const float HintSeconds = 8f;

        public static NpcFreeTalk Instance { get; private set; }

        private NpcAgentClient client;
        private readonly List<NpcAgentClient.TurnDto> history = new List<NpcAgentClient.TurnDto>();

        private string npcId, speakerName;
        private string[] openers = new string[0];
        private bool offerAvailable;    // service answered for this NPC
        private bool open, asking;
        private float hintUntil;

        private Canvas canvas;
        private GameObject panel, hint;
        private TextMeshProUGUI hintText, speakerLabel, bodyLabel, statusLabel;
        private TMP_InputField input;
        private readonly List<Button> openerButtons = new List<Button>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Bootstrap()
        {
            if (Instance == null) new GameObject("[NpcFreeTalk]").AddComponent<NpcFreeTalk>();
        }

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            client = gameObject.AddComponent<NpcAgentClient>();
            Build();
            SetOpen(false);
            hint.SetActive(false);
        }

        private void OnEnable()
        {
            if (DialogueManager.Instance == null) return;
            DialogueManager.Instance.OnDialogueCompletedId += OnSpokeTo;
            DialogueManager.Instance.OnDialogueEnded += OnDialogueClosed;
        }

        private void OnDisable()
        {
            if (DialogueManager.Instance == null) return;
            DialogueManager.Instance.OnDialogueCompletedId -= OnSpokeTo;
            DialogueManager.Instance.OnDialogueEnded -= OnDialogueClosed;
        }

        private void Start()
        {
            // DialogueManager may boot after this component; rebind once it exists.
            StartCoroutine(BindWhenReady());
        }

        private IEnumerator BindWhenReady()
        {
            while (DialogueManager.Instance == null) yield return null;
            OnDisable();
            OnEnable();
        }

        // --- entry point ------------------------------------------------

        private void OnSpokeTo(string id)
        {
            if (string.IsNullOrEmpty(id) || id == npcId) return;
            npcId = id;
            offerAvailable = false;
            history.Clear();
            StartCoroutine(FetchInfo(id));
        }

        private IEnumerator FetchInfo(string id)
        {
            string language = LocalizationManager.Current == Language.Chinese ? "zh" : "en";
            yield return client.FetchNpcInfo(id, language,
                info =>
                {
                    // A different NPC may have been spoken to while this was in flight.
                    if (id != npcId) return;
                    speakerName = info.name;
                    openers = info.openers ?? new string[0];
                    offerAvailable = true;
                },
                _ => { /* service down or NPC not configured — stay silent */ });
        }

        private void OnDialogueClosed()
        {
            if (!offerAvailable || open) return;
            hintText.text = LocalizationManager.Current == Language.Chinese
                ? $"按 <color=#F0CE77>T</color> 与{speakerName}继续交谈"
                : $"Press <color=#F0CE77>T</color> to keep talking with {speakerName}";
            hint.SetActive(true);
            hintUntil = Time.unscaledTime + HintSeconds;
        }

        private void Update()
        {
            if (hint.activeSelf && Time.unscaledTime > hintUntil && !open) hint.SetActive(false);

            if (!open)
            {
                if (offerAvailable && hint.activeSelf && Input.GetKeyDown(OpenKey)) SetOpen(true);
                return;
            }

            if (Input.GetKeyDown(KeyCode.Escape)) { SetOpen(false); return; }

            if (!asking && Input.GetKeyDown(KeyCode.Return) && input.isFocused)
            {
                string text = input.text.Trim();
                if (text.Length > 0) Ask(text);
            }
        }

        private void SetOpen(bool value)
        {
            open = value;
            panel.SetActive(value);
            hint.SetActive(false);

            if (GameManager.Instance != null)
                GameManager.Instance.SetState(value ? GameState.InDialogue : GameState.Exploring);

            if (!value) return;

            speakerLabel.text = speakerName;
            bodyLabel.text = string.Empty;
            statusLabel.text = string.Empty;
            RefreshOpeners();
            input.text = string.Empty;
            input.ActivateInputField();
        }

        private void RefreshOpeners()
        {
            for (int i = 0; i < openerButtons.Count; i++)
            {
                bool used = i < openers.Length;
                openerButtons[i].gameObject.SetActive(used);
                if (!used) continue;

                string question = openers[i];
                openerButtons[i].GetComponentInChildren<TextMeshProUGUI>().text = question;
                openerButtons[i].onClick.RemoveAllListeners();
                openerButtons[i].onClick.AddListener(() => { if (!asking) Ask(question); });
            }
        }

        // --- asking -----------------------------------------------------

        private void Ask(string question) => StartCoroutine(AskRoutine(question));

        private IEnumerator AskRoutine(string question)
        {
            asking = true;
            input.text = string.Empty;
            input.DeactivateInputField();

            bool zh = LocalizationManager.Current == Language.Chinese;
            bodyLabel.text = zh ? $"<color=#7E96CC>旅者：{question}</color>\n\n" : $"<color=#7E96CC>Traveler: {question}</color>\n\n";
            statusLabel.text = "…";

            string prefix = bodyLabel.text;
            string shown = string.Empty;

            NpcAgentClient.PlayerStateDto state = ChronoAgentBridge.Collect(npcId);

            yield return client.ChatStream(
                npcId, question, state, history,
                // Deltas drive the text directly — the game already reads dialogue
                // a character at a time, so streaming *is* the typewriter here.
                onDelta: delta => { shown += delta; bodyLabel.text = prefix + shown; },
                // Mandatory: a guardrail retracting what is already on screen.
                onReplace: text => { shown = text; bodyLabel.text = prefix + shown; },
                onDone: reply =>
                {
                    bodyLabel.text = prefix + reply.text;
                    statusLabel.text = reply.IsFallback
                        ? (zh ? "（史官一时无言）" : "(the historian says nothing)")
                        : string.Empty;

                    history.Add(new NpcAgentClient.TurnDto("user", question));
                    history.Add(new NpcAgentClient.TurnDto("assistant", reply.text));
                    if (history.Count > 12) history.RemoveRange(0, history.Count - 12);
                },
                onError: error =>
                {
                    bodyLabel.text = prefix + (zh ? "……" : "…");
                    statusLabel.text = zh ? "（连接不上）" : "(cannot reach the service)";
                    Debug.LogWarning($"[NpcFreeTalk] {error}");
                },
                freeForm: true);

            asking = false;
            input.ActivateInputField();
        }

        // --- UI ---------------------------------------------------------

        private void Build()
        {
            var canvasGo = new GameObject("FreeTalkCanvas");
            canvasGo.transform.SetParent(transform, false);
            canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = SortOrder;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            canvasGo.AddComponent<GraphicRaycaster>();

            BuildHint(canvasGo.transform);
            BuildPanel(canvasGo.transform);
        }

        private void BuildHint(Transform parent)
        {
            hint = new GameObject("Hint");
            hint.transform.SetParent(parent, false);
            var rt = hint.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(.5f, 0f); rt.anchorMax = new Vector2(.5f, 0f);
            rt.pivot = new Vector2(.5f, 0f);
            rt.anchoredPosition = new Vector2(0, 44);
            rt.sizeDelta = new Vector2(900, 44);

            hintText = UIFactory.CreateText("HintText", hint.transform, 26,
                TextAlignmentOptions.Center, UITheme.TextDim);
            Stretch(hintText.rectTransform);
        }

        private void BuildPanel(Transform parent)
        {
            panel = new GameObject("Panel");
            panel.transform.SetParent(parent, false);
            var rt = panel.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(.5f, 0f); rt.anchorMax = new Vector2(.5f, 0f);
            rt.pivot = new Vector2(.5f, 0f);
            rt.anchoredPosition = new Vector2(0, 40);
            rt.sizeDelta = new Vector2(1500, 470);

            Image fill = UIFactory.CreateImage("Fill", panel.transform, UITheme.WindowFill);
            Stretch(fill.rectTransform);

            speakerLabel = UIFactory.CreateText("Speaker", panel.transform, 30,
                TextAlignmentOptions.TopLeft, UITheme.FrameGold);
            Place(speakerLabel.rectTransform, 28, -18, 700, 40);

            bodyLabel = UIFactory.CreateText("Body", panel.transform, 27,
                TextAlignmentOptions.TopLeft, UITheme.TextMain);
            Place(bodyLabel.rectTransform, 28, -62, 1444, 210);

            statusLabel = UIFactory.CreateText("Status", panel.transform, 22,
                TextAlignmentOptions.TopRight, UITheme.TextDim);
            Place(statusLabel.rectTransform, 28, -18, 1444, 34);

            BuildOpeners();
            BuildInput();

            var close = UIFactory.CreateText("CloseHint", panel.transform, 21,
                TextAlignmentOptions.BottomRight, UITheme.TextDim);
            close.text = "ESC";
            Place(close.rectTransform, 28, -430, 1444, 30);
        }

        private void BuildOpeners()
        {
            for (int i = 0; i < 4; i++)
            {
                var go = new GameObject($"Opener{i}");
                go.transform.SetParent(panel.transform, false);
                var image = go.AddComponent<Image>();
                image.color = UITheme.WindowFillTop;
                var button = go.AddComponent<Button>();
                button.targetGraphic = image;

                var label = UIFactory.CreateText("Label", go.transform, 22,
                    TextAlignmentOptions.Left, UITheme.TextDim);
                var lrt = label.rectTransform;
                lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one;
                lrt.offsetMin = new Vector2(14, 0); lrt.offsetMax = new Vector2(-14, 0);

                Place(go.GetComponent<RectTransform>(), 28 + (i % 2) * 730,
                      -288 - (i / 2) * 46, 710, 40);
                openerButtons.Add(button);
            }
        }

        private void BuildInput()
        {
            var go = new GameObject("Input");
            go.transform.SetParent(panel.transform, false);
            var image = go.AddComponent<Image>();   // brings a RectTransform with it
            image.color = UITheme.BarTrack;
            Place(go.GetComponent<RectTransform>(), 28, -388, 1444, 46);

            input = go.AddComponent<TMP_InputField>();
            input.image = image;
            input.lineType = TMP_InputField.LineType.SingleLine;
            input.characterLimit = 200;

            var area = new GameObject("TextArea");
            area.transform.SetParent(go.transform, false);
            var art = area.AddComponent<RectTransform>();
            art.anchorMin = Vector2.zero; art.anchorMax = Vector2.one;
            art.offsetMin = new Vector2(14, 0); art.offsetMax = new Vector2(-14, 0);
            area.AddComponent<RectMask2D>();

            var text = UIFactory.CreateText("Text", area.transform, 24,
                TextAlignmentOptions.Left, UITheme.TextMain);
            text.raycastTarget = true;
            Stretch(text.rectTransform);

            var placeholder = UIFactory.CreateText("Placeholder", area.transform, 24,
                TextAlignmentOptions.Left, UITheme.TextDim);
            placeholder.text = LocalizationManager.Current == Language.Chinese
                ? "问些什么……（回车递话）" : "Ask something… (Enter to send)";
            Stretch(placeholder.rectTransform);

            input.textViewport = art;
            input.textComponent = text;
            input.placeholder = placeholder;
        }

        private static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        /// <summary>Top-left anchored placement, in the panel's own space.</summary>
        private static void Place(RectTransform rt, float x, float y, float w, float h)
        {
            if (rt == null) return;
            rt.anchorMin = new Vector2(0, 1); rt.anchorMax = new Vector2(0, 1);
            rt.pivot = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(x, y);
            rt.sizeDelta = new Vector2(w, h);
        }
    }
}

using System.Collections.Generic;
using ChronoTraveler.Collectibles;
using ChronoTraveler.Core;
using ChronoTraveler.Quest;
using UnityEngine;

namespace ChronoTraveler.Agent
{
    /// <summary>
    /// Reads the live game state into the shape the agent service expects. This
    /// is the only file here that knows ChronoTraveler exists; keeping it apart
    /// means <see cref="NpcAgentClient"/> stays a plain HTTP client.
    ///
    /// Two things it deliberately does not do:
    ///
    /// It reports only what is relevant to the NPC being spoken to — whether
    /// *this* NPC has been met, and the encounters named by the *current* quest.
    /// SaveManager exposes membership tests rather than the whole set, and that
    /// turns out to be the better shape anyway: an NPC has no business knowing
    /// the player's progress in an era they have never mentioned.
    ///
    /// It sends no exact quest wording beyond the stage the player is on. The
    /// service pares state down further before it reaches the model — see the
    /// note on state injection in persona.py — but there is no reason to ship
    /// what will only be discarded.
    /// </summary>
    public static class ChronoAgentBridge
    {
        /// <summary>Snapshot the state an NPC is allowed to perceive.</summary>
        public static NpcAgentClient.PlayerStateDto Collect(string npcId)
        {
            var dto = new NpcAgentClient.PlayerStateDto
            {
                current_map = SceneFlowManager.Instance != null &&
                              !string.IsNullOrEmpty(SceneFlowManager.Instance.CurrentMapScene)
                    ? SceneFlowManager.Instance.CurrentMapScene
                    : "20_China",
                language = LocalizationManager.Current == Language.Chinese ? "zh" : "en"
            };

            if (MemoryProgress.Instance != null)
                dto.memory_progress = Mathf.Clamp01(MemoryProgress.Instance.GetProgress(dto.current_map));

            if (GameManager.Instance != null)
            {
                dto.correct_answers = GameManager.Instance.CorrectAnswers;
                dto.total_answers = GameManager.Instance.TotalAnswers;
            }

            if (EraProgress.Instance != null)
                dto.earned_tokens = EraProgress.Instance.GetState().ToArray();

            if (SaveManager.Instance != null && SaveManager.Instance.IsNPCTalked(npcId))
                dto.talked_npcs = new[] { npcId };

            ApplyQuest(dto);
            return dto;
        }

        private static void ApplyQuest(NpcAgentClient.PlayerStateDto dto)
        {
            QuestManager quests = QuestManager.Instance;
            if (quests == null) return;

            QuestRuntime runtime = quests.CurrentQuest();
            QuestDefinition def = quests.CurrentDef();
            if (runtime == null || def == null) return;

            var stage = runtime.stageIndex >= 0 && runtime.stageIndex < def.stages.Count
                ? def.stages[runtime.stageIndex]
                : null;

            var quest = new NpcAgentClient.QuestDto
            {
                quest_id = runtime.questId,
                stage_index = runtime.stageIndex,
                stage_description = stage != null ? stage.description : string.Empty
            };

            // A stage can hold several objectives; the first is the one the
            // tracker shows, so it is the one an NPC would speak about.
            if (stage != null && stage.objectives.Count > 0)
            {
                QuestObjective objective = stage.objectives[0];
                quest.objective_label = objective.label;
                quest.required = Mathf.Max(1, objective.required);
                quest.progress = runtime.progress != null && runtime.progress.Count > 0
                    ? runtime.progress[0]
                    : 0;
            }

            dto.quest = quest;
            ApplyFragments(dto, def);
            ApplyPurged(dto, def);
        }

        /// <summary>
        /// Fragment counts come from the collection manager rather than the quest
        /// runtime, because quest progress only tracks the stage the player is
        /// currently on — once they move past collecting, it stops reflecting
        /// what they hold.
        /// </summary>
        private static void ApplyFragments(NpcAgentClient.PlayerStateDto dto, QuestDefinition def)
        {
            string prefix = null;
            int required = 0;
            foreach (QuestStage stage in def.stages)
            {
                foreach (QuestObjective objective in stage.objectives)
                {
                    if (objective.type != ObjectiveType.Collect) continue;
                    prefix = objective.targetId;      // e.g. "frag_china"
                    required = Mathf.Max(1, objective.required);
                    break;
                }
                if (prefix != null) break;
            }
            if (prefix == null || CollectionManager.Instance == null) return;

            int held = 0;
            foreach (string id in CollectionManager.Instance.GetCollectedIds())
                if (!string.IsNullOrEmpty(id) && id.StartsWith(prefix)) held++;

            dto.fragments_collected = held;
            dto.fragments_required = required;
        }

        private static void ApplyPurged(NpcAgentClient.PlayerStateDto dto, QuestDefinition def)
        {
            if (SaveManager.Instance == null) return;

            var purged = new List<string>();
            foreach (QuestStage stage in def.stages)
            {
                foreach (QuestObjective objective in stage.objectives)
                {
                    if (objective.type != ObjectiveType.Defeat) continue;
                    if (string.IsNullOrEmpty(objective.targetId) || objective.targetId == "any") continue;
                    if (SaveManager.Instance.IsEncounterPurged(objective.targetId))
                        purged.Add(objective.targetId);
                }
            }
            if (purged.Count > 0) dto.purged_encounters = purged.ToArray();
        }
    }
}

import { createInterface } from "node:readline";
import { readFileSync } from "node:fs";

import { profileDeck } from "../../riggedroyale/src/lib/luck-engine/archetype";
import {
  getCardDb,
  resolveCard,
  type ResolvedCard,
} from "../../riggedroyale/src/lib/luck-engine/card-db";
import {
  addBattleToEmpiricalMatchupRawCounts,
  buildEmpiricalMatchups,
  deckPlanKey,
  deserializeEmpiricalMatchups,
  empiricalBlendedDelta,
  emptyEmpiricalMatchupRawCounts,
  finalizeEmpiricalMatchupRawCounts,
  serializeEmpiricalMatchups,
  spellAnswerKey,
  type EmpiricalMatchups,
  type MatchupContext,
  type MatchupOutcome,
} from "../../riggedroyale/src/lib/luck-engine/empirical-matchup";
import type { RawBattle } from "../../riggedroyale/src/lib/luck-engine/types";
import { winProbability } from "../../riggedroyale/src/lib/luck-engine/win-probability";

interface BuildMatrixRow {
  playerTag?: string;
  player_tag?: string;
  battle?: RawBattle;
  raw?: RawBattle;
}

interface ScoreRow {
  segment?: string;
  bucket?: string;
  team_card_ids?: number[];
  opponent_card_ids?: number[];
  win?: boolean;
}

const db = getCardDb();

function usage(): never {
  throw new Error(
    [
      "Usage:",
      "  empirical-prior-helper.ts build-matrix < rows.jsonl",
      "  empirical-prior-helper.ts build-matrix-from-records < prepared-rows.jsonl",
      "  empirical-prior-helper.ts score-records --matrix matrix.json < rows.jsonl",
    ].join("\n"),
  );
}

function parseArgs(): { command: string; matrixPath?: string } {
  const [, , command, ...rest] = process.argv;
  if (!command) usage();
  let matrixPath: string | undefined;
  for (let index = 0; index < rest.length; index += 1) {
    const value = rest[index];
    if (value === "--matrix") {
      matrixPath = rest[index + 1];
      index += 1;
    } else {
      usage();
    }
  }
  return { command, matrixPath };
}

async function eachInputLine(
  callback: (value: unknown) => void,
): Promise<number> {
  const input = createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });
  let rows = 0;
  for await (const line of input) {
    const text = line.trim();
    if (!text) continue;
    callback(JSON.parse(text));
    rows += 1;
  }
  return rows;
}

function clampProbability(value: number): number {
  if (!Number.isFinite(value)) return 0.5;
  return Math.min(0.999, Math.max(0.001, value));
}

function bucketFromSegment(segment: string | undefined): string {
  if (!segment) return "unknown";
  if (segment === "ranked") return "ranked:any";
  if (segment.startsWith("ranked:league-")) return segment;
  if (segment === "ranked:unknown") return "ranked:unknown";
  if (segment === "ladder:0-4999") return "ladder:low";
  if (segment === "ladder:7000-8999") return "ladder:mid";
  if (segment === "ladder:9000-11999") return "ladder:high";
  if (segment === "ladder:12000-99998") return "ladder:top";
  if (segment.startsWith("ladder:top-")) return "ladder:top";
  return "unknown";
}

function resolveDeck(cardIds: number[] | undefined): ResolvedCard[] {
  return (cardIds ?? [])
    .slice(0, 8)
    .filter((id) => Number.isFinite(id))
    .map((id) =>
      resolveCard(db, {
        id: Number(id),
        name: String(id),
      }),
    );
}

function contextFor(row: ScoreRow): MatchupContext {
  const team = resolveDeck(row.team_card_ids);
  const opponent = resolveDeck(row.opponent_card_ids);
  const archA = profileDeck(team).archetype;
  const archB = profileDeck(opponent).archetype;
  const typeA = deckPlanKey(team, archA);
  const typeB = deckPlanKey(opponent, archB);
  return {
    bucket: row.bucket ?? bucketFromSegment(row.segment),
    archA,
    archB,
    typeA,
    typeB,
    spellA: spellAnswerKey(team, typeB),
    spellB: spellAnswerKey(opponent, typeA),
  };
}

function matrixKey(prefix: string, a: string, b: string): string {
  return prefix ? `${prefix}#${a}>${b}` : `${a}>${b}`;
}

function coverageFor(matrix: EmpiricalMatchups, ctx: MatchupContext) {
  return {
    archGlobal: matrix.archGlobal.has(matrixKey("", ctx.archA, ctx.archB)),
    archBucket: matrix.archBucket.has(matrixKey(ctx.bucket, ctx.archA, ctx.archB)),
    planGlobal: matrix.planGlobal.has(matrixKey("", ctx.typeA, ctx.typeB)),
    planBucket: matrix.planBucket.has(matrixKey(ctx.bucket, ctx.typeA, ctx.typeB)),
    spellAnswerGlobal:
      ctx.spellA !== undefined &&
      ctx.spellB !== undefined &&
      matrix.spellAnswerGlobal.has(matrixKey("", ctx.spellA, ctx.spellB)),
  };
}

async function buildMatrix(): Promise<void> {
  const raw = emptyEmpiricalMatchupRawCounts();
  await eachInputLine((value) => {
    const row = value as BuildMatrixRow;
    const playerTag = row.playerTag ?? row.player_tag;
    const battle = row.battle ?? row.raw;
    if (!playerTag || !battle) return;
    addBattleToEmpiricalMatchupRawCounts(raw, db, playerTag, battle);
  });
  const matrix = finalizeEmpiricalMatchupRawCounts(raw);
  process.stdout.write(JSON.stringify(serializeEmpiricalMatchups(matrix)));
}

async function buildMatrixFromRecords(): Promise<void> {
  const outcomes: MatchupOutcome[] = [];
  await eachInputLine((value) => {
    const row = value as ScoreRow;
    const ctx = contextFor(row);
    outcomes.push({
      ...ctx,
      result: row.win ? 1 : 0,
    });
  });
  const matrix = buildEmpiricalMatchups(outcomes);
  process.stdout.write(JSON.stringify(serializeEmpiricalMatchups(matrix)));
}

async function scoreRecords(matrixPath: string | undefined): Promise<void> {
  if (!matrixPath) usage();
  const matrix = deserializeEmpiricalMatchups(
    JSON.parse(readFileSync(matrixPath, "utf8")),
  );
  await eachInputLine((value) => {
    const row = value as ScoreRow;
    const ctx = contextFor(row);
    const empiricalDelta = empiricalBlendedDelta(matrix, 0, ctx);
    const prior = clampProbability(winProbability(empiricalDelta, 0, null));
    process.stdout.write(
      `${JSON.stringify({
        prior,
        coverage: coverageFor(matrix, ctx),
        context: ctx,
      })}\n`,
    );
  });
}

async function main() {
  const { command, matrixPath } = parseArgs();
  if (command === "build-matrix") {
    await buildMatrix();
    return;
  }
  if (command === "build-matrix-from-records") {
    await buildMatrixFromRecords();
    return;
  }
  if (command === "score-records") {
    await scoreRecords(matrixPath);
    return;
  }
  usage();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

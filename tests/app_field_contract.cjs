/* Pure-function contracts for the field display. No DOM, no model. */
'use strict';
const assert = require('assert');
const path = require('path');

require(path.join(__dirname, '..', 'looking_glass', 'app_static', 'app_field.js'));
const field = globalThis.MLGField;
assert.strictEqual(typeof field.create, 'function');
const ROWS = field.ROWS;

function token(id, probability) { return {token_id: id, text: 't' + id, probability}; }
function count(rows, id) { return rows.filter((entry) => entry && entry.token_id === id).length; }

// Rows are area: a token's share of rows tracks its probability, and uncovered mass stays empty.
{
  const rows = field.allocateRows([token(1, 0.5), token(2, 0.25), token(3, 0.05)], ROWS);
  assert.strictEqual(rows.length, ROWS);
  assert.strictEqual(count(rows, 1), 8);
  assert.strictEqual(count(rows, 2), 4);
  assert.strictEqual(count(rows, 3), 1);
  assert.strictEqual(rows.filter((entry) => entry === null).length, 3);
  assert.ok(rows.indexOf(null) > rows.lastIndexOf(rows[0]), 'empty rows sit below the words');
}
// A confident distribution fills the column; a flat one leaves it almost empty but never blank.
assert.strictEqual(count(field.allocateRows([token(1, 1)], ROWS), 1), ROWS);
{
  const flat = field.allocateRows([token(1, 0.004), token(2, 0.003)], ROWS);
  assert.strictEqual(count(flat, 1), 1);
  assert.strictEqual(flat.filter((entry) => entry === null).length, ROWS - 1);
}
assert.deepStrictEqual(field.allocateRows([], 4), [null, null, null, null]);
// Over many random distributions: fixed length, no token outranks a likelier one, total matches covered mass.
for (let trial = 0; trial < 500; trial++) {
  let left = 1;
  const tokens = [];
  for (let id = 0; id < 10; id++) {
    const share = left * field.hash(trial, id, 3) * 0.6;
    tokens.push(token(id, share)); left -= share;
  }
  tokens.sort((a, b) => b.probability - a.probability);
  const rows = field.allocateRows(tokens, ROWS), counts = tokens.map((entry) => count(rows, entry.token_id));
  assert.strictEqual(rows.length, ROWS);
  for (let i = 1; i < counts.length; i++) assert.ok(counts[i] <= counts[i - 1], 'rows must not increase down the ranking');
  const covered = tokens.reduce((sum, entry) => sum + entry.probability, 0);
  const filled = counts.reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(filled - Math.max(1, Math.round(covered * ROWS))) <= 1, 'filled rows match covered probability');
}

// Blending: the ends reproduce each layer, the middle is the average, and order is by probability.
{
  const lower = [token(1, 0.6), token(2, 0.2)], upper = [token(2, 0.5), token(3, 0.3)];
  assert.deepStrictEqual(field.blendTokens(lower, upper, 0).map((entry) => entry.token_id), [1, 2]);
  assert.deepStrictEqual(field.blendTokens(lower, upper, 1).map((entry) => entry.token_id), [2, 3]);
  const middle = field.blendTokens(lower, upper, 0.5);
  assert.deepStrictEqual(middle.map((entry) => entry.token_id), [2, 1, 3]);
  assert.ok(Math.abs(middle[0].probability - 0.35) < 1e-12 && Math.abs(middle[1].probability - 0.3) < 1e-12);
  const total = middle.reduce((sum, entry) => sum + entry.probability, 0);
  assert.ok(Math.abs(total - 0.8) < 1e-12, 'blended mass is the average of the two layers');
  assert.strictEqual(lower[0].probability, 0.6, 'inputs are not modified');
}

// Entropy share is clamped to [0, 1].
assert.strictEqual(field.normalizedEntropy(0, 50257), 0);
assert.strictEqual(field.normalizedEntropy(Math.log(50257), 50257), 1);
assert.strictEqual(field.normalizedEntropy(99, 50257), 1);
assert.ok(field.normalizedEntropy(3, 50257) > 0.2 && field.normalizedEntropy(3, 50257) < 0.4);

// Tiling fills exactly the column and never drops the label.
assert.strictEqual(field.tile('·Paris', 12), '·Paris ·Pari');
assert.strictEqual(field.tile('', 5).length, 5);
assert.strictEqual(field.tile('abcdefghijkl', 7), 'abcdefg');

// Columns shrink to fit, within bounds.
assert.strictEqual(field.columnChars(2000, 6, 5, 7.2), 12);
assert.strictEqual(field.columnChars(400, 6, 40, 7.2), 7);
{
  const chars = field.columnChars(890, 6, 11, 7.2);
  assert.ok(chars >= 7 && chars <= 12 && (6 + 11 * (chars + 2)) * 7.2 <= 890, 'eleven columns fit in 890px');
}

// The resolution layer is the first one where the answer ranks first.
{
  const layers = [3, 2, 1, 1, 2].map((rank, index) => ({layer_index: index, cells: [{position: 4, answer_rank: rank}]}));
  assert.strictEqual(field.resolvedAt(layers, 4), 2);
  assert.strictEqual(field.resolvedAt(layers.slice(0, 2), 4), -1);
  assert.strictEqual(field.resolvedAt(layers, 9), -1);
}

// The pseudo-random source is deterministic and stays in [0, 1).
assert.strictEqual(field.hash(3, 5, 7), field.hash(3, 5, 7));
for (let n = 0; n < 2000; n++) { const value = field.hash(n, n * 7, n * 13); assert.ok(value >= 0 && value < 1); }

console.log('app_field_contract: PASS');

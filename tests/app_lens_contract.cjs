/* Pure-function contracts for the lens view's encodings. No DOM, no model. */
'use strict';
const assert = require('assert');
const path = require('path');

require(path.join(__dirname, '..', 'looking_glass', 'app_static', 'app_lens.js'));
const lens = globalThis.MLGLens;
assert.strictEqual(typeof lens.create, 'function');

// Whitespace stays visible and is never silently trimmed away.
assert.deepStrictEqual(lens.tokenParts(' Paris'), {lead: true, body: 'Paris', trail: false});
assert.deepStrictEqual(lens.tokenParts('Paris'), {lead: false, body: 'Paris', trail: false});
assert.deepStrictEqual(lens.tokenParts('\n'), {lead: false, body: '⏎', trail: false});
assert.deepStrictEqual(lens.tokenParts('  '), {lead: false, body: '␣␣', trail: false});
assert.deepStrictEqual(lens.tokenParts(''), {lead: false, body: '(empty)', trail: false});

// Percentages never round a nonzero probability to "0%".
assert.strictEqual(lens.percent(0.3649), '36%');
assert.strictEqual(lens.percent(0.064), '6.4%');
assert.strictEqual(lens.percent(0.00001), '<0.1%');

// Rank magnitude: #1 is the maximum, the last rank is the minimum, and it is monotone.
const vocabulary = 50257;
assert.strictEqual(lens.rankMagnitude(1, vocabulary), 1);
assert.ok(lens.rankMagnitude(vocabulary, vocabulary) < 0.01);
let previous = Infinity;
for (const rank of [1, 2, 10, 100, 1000, 10000, vocabulary]) {
  const value = lens.rankMagnitude(rank, vocabulary);
  assert.ok(value < previous || rank === 1, 'rank magnitude must fall as rank grows');
  previous = value;
}

// The ramp is one hue that only gets darker, and cell text keeps WCAG AA contrast on every step.
function luminance(hex) {
  const channel = (offset) => {
    const value = parseInt(hex.slice(offset, offset + 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}
function contrast(a, b) {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (light + 0.05) / (dark + 0.05);
}
let lightness = Infinity, steps = new Set();
for (let step = 0; step <= 100; step++) {
  const fill = lens.rampColor(step / 100);
  steps.add(fill);
  assert.ok(luminance(fill) <= lightness, 'ramp must darken monotonically');
  lightness = luminance(fill);
  assert.ok(contrast(fill, lens.inkOn(fill)) >= 4.5, `text contrast on ${fill}`);
}
assert.strictEqual(steps.size, 13);
assert.strictEqual(lens.rampColor(-1), lens.rampColor(0));
assert.strictEqual(lens.rampColor(2), lens.rampColor(1));

console.log('app_lens_contract: PASS');

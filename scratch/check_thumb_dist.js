import fs from 'fs';

try {
  const fileContent = fs.readFileSync('asl_alphabet.json', 'utf8');
  const samples = JSON.parse(fileContent);

  const getDistances = (label) => {
    const matching = samples.filter(s => s.label === label);
    console.log(`=== ${label} (${matching.length} samples) ===`);
    matching.slice(0, 5).forEach((sample, idx) => {
      const features = sample.features;
      const lm = [];
      for (let i = 0; i < 21; i++) {
        lm.push({
          x: features[i * 3],
          y: features[i * 3 + 1],
          z: features[i * 3 + 2]
        });
      }
      const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      const d4_9 = dist(lm[4], lm[9]);
      console.log(`Sample ${idx}: lm[4].x = ${lm[4].x.toFixed(3)}, d(4,9) = ${d4_9.toFixed(3)}`);
    });
  };

  getDistances('B');
  getDistances('Y');
  getDistances('D');
  getDistances('L');
  getDistances('G');
} catch (e) {
  console.error(e);
}

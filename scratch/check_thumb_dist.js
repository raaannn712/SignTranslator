import fs from 'fs';

try {
  const fileContent = fs.readFileSync('asl_alphabet.json', 'utf8');
  const samples = JSON.parse(fileContent);

  const getXs = (label) => {
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
      console.log(`Sample ${idx}: lm[8].x = ${lm[8].x.toFixed(3)}, lm[12].x = ${lm[12].x.toFixed(3)}, diff(8-12) = ${(lm[8].x - lm[12].x).toFixed(3)}`);
    });
  };

  getXs('R');
  getXs('U');
  getXs('V');
} catch (e) {
  console.error(e);
}

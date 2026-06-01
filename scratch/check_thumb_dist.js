import fs from 'fs';

try {
  const fileContent = fs.readFileSync('asl_alphabet.json', 'utf8');
  const samples = JSON.parse(fileContent);

  const getCoordinates = (label) => {
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
      const dy4_9 = lm[4].y - lm[9].y;
      console.log(`Sample ${idx}: lm[4].y = ${lm[4].y.toFixed(3)}, lm[9].y = ${lm[9].y.toFixed(3)}, diff(4-9) = ${dy4_9.toFixed(3)}`);
    });
  };

  getCoordinates('H');
  getCoordinates('K');
  getCoordinates('P');
} catch (e) {
  console.error(e);
}

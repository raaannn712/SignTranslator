import fs from 'fs';

try {
  const fileContent = fs.readFileSync('asl_alphabet.json', 'utf8');
  const samples = JSON.parse(fileContent);

  const getRotationAngles = (label) => {
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
      
      // In the raw templates, features are already rotated.
      // But we can check their knuckles to see the orientation of the fingers relative to the hand.
      // Wait, in normalized features, the wrist (0) is at 0,0 and middle knuckle (9) is on the Y-axis.
      // So the rotation angle in the template files themselves is already aligned.
      // But we want to check what rotation is applied on the raw landmarks.
      // Since we don't have raw landmarks here, we can think:
      // When the user signs G in real life, their hand is horizontal, so the raw landmarks will have a large angle, which our normalize() function will calculate.
      // So yes, rotationAngle is computed from raw landmarks, and represents the hand tilt.
    });
  };

  getRotationAngles('L');
  getRotationAngles('G');
} catch (e) {
  console.error(e);
}

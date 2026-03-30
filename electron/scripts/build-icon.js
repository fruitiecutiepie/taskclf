const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const path = require('path');

async function buildIcon() {
  const svgPath = path.resolve(__dirname, '../../assets/taskclf-icon.svg');
  const outDir = path.resolve(__dirname, '../build');
  const outPath = path.join(outDir, 'icon.png');

  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const svg = fs.readFileSync(svgPath);
  const resvg = new Resvg(svg, {
    fitTo: {
      mode: 'width',
      value: 1024,
    },
  });

  const pngData = resvg.render().asPng();
  fs.writeFileSync(outPath, pngData);
  console.log(`Successfully built icon to ${outPath}`);
}

buildIcon().catch(err => {
  console.error(err);
  process.exit(1);
});

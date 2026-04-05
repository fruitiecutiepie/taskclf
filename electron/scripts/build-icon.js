const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const path = require('path');

function renderSvgToPng(svgPath, outPath, width) {
  const svg = fs.readFileSync(svgPath);
  const resvg = new Resvg(svg, {
    fitTo: {
      mode: 'width',
      value: width,
    },
  });
  fs.writeFileSync(outPath, resvg.render().asPng());
}

async function buildIcon() {
  const outDir = path.resolve(__dirname, '../build');

  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const builds = [
    {
      svgPath: path.resolve(__dirname, '../../assets/taskclf-icon.svg'),
      outPath: path.join(outDir, 'icon.png'),
      width: 1024,
    },
    {
      svgPath: path.resolve(__dirname, '../../assets/taskclf-tray-template.svg'),
      outPath: path.join(outDir, 'trayTemplate.png'),
      width: 16,
    },
    {
      svgPath: path.resolve(__dirname, '../../assets/taskclf-tray-template.svg'),
      outPath: path.join(outDir, 'trayTemplate@2x.png'),
      width: 32,
    },
  ];

  for (const build of builds) {
    renderSvgToPng(build.svgPath, build.outPath, build.width);
    console.log(`Successfully built icon to ${build.outPath}`);
  }
}

buildIcon().catch(err => {
  console.error(err);
  process.exit(1);
});

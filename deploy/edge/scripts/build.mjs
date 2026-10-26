import {readFileSync,writeFileSync,mkdirSync,rmSync,cpSync} from 'node:fs';
const data=readFileSync('worker/public-data.json','utf8'),report=readFileSync('worker/report.json','utf8'),page=readFileSync('worker/page.html','utf8');
rmSync('dist',{recursive:true,force:true});mkdirSync('dist/server',{recursive:true});mkdirSync('dist/.openai',{recursive:true});
writeFileSync('dist/server/index.js',`const DATA=${data};\nconst REPORT=${report};\nconst PAGE=${JSON.stringify(page)};\n`+readFileSync('worker/app.mjs','utf8'));
cpSync('.openai/hosting.json','dist/.openai/hosting.json');
console.log('Built public Worker with metadata features, trained embeddings and aggregate report. No historical user interactions.');

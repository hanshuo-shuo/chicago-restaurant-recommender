import {createServer} from 'node:http';
import {readFileSync} from 'node:fs';
import {localDB} from './local-db.mjs';
const source=readFileSync('dist/server/index.js','utf8');
const worker=(await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
const DB=localDB('.preview.sqlite');
const server=createServer(async(req,res)=>{try{
  const chunks=[];for await(const chunk of req){chunks.push(chunk);if(chunks.reduce((s,c)=>s+c.length,0)>16384){res.writeHead(413);res.end('Request too large');return;}}
  const request=new Request('http://127.0.0.1:8777'+req.url,{method:req.method,headers:req.headers,...(req.method==='POST'?{body:Buffer.concat(chunks)}:{})});
  const result=await worker.fetch(request,{DB},{});res.writeHead(result.status,Object.fromEntries(result.headers));res.end(Buffer.from(await result.arrayBuffer()));
}catch{res.writeHead(500);res.end('Preview unavailable');}});
server.listen(8777,'127.0.0.1',()=>console.log('Public-site preview: http://127.0.0.1:8777'));
process.on('SIGINT',()=>{server.close();DB.close();process.exit(0);});

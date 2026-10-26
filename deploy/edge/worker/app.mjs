const norm=v=>{const length=Math.sqrt(v.reduce((s,x)=>s+x*x,0));return v.map(x=>x/Math.max(length,1e-12));};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const minmax=v=>{const lo=Math.min(...v),span=Math.max(...v)-lo;return v.map(x=>span>1e-12?(x-lo)/span:0);};
const itemIndex=new Map(DATA.items.map((item,i)=>[item.item_id,i]));
const allowedCuisines=new Set(DATA.items.map(item=>item.cuisine));
const WEEK=7*24*3600*1000;
class Invalid extends Error{}
function keys(value,allowed){if(!value||typeof value!=='object'||Array.isArray(value))throw new Invalid('Provide a JSON object.');if(Object.keys(value).some(k=>!allowed.includes(k)))throw new Invalid('Unsupported fields. Historical identities are not accepted.');}
function session(value){if(typeof value!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value))throw new Invalid('Use an anonymous browser session UUID v4.');return value.toLowerCase();}
function validatePreferences(p={}){
  keys(p,['cuisines','budget','open_after_8pm','seed_item_ids','exclude_seen']);
  for(const [key,known] of [['cuisines',allowedCuisines],['seed_item_ids',new Set(itemIndex.keys())]]){
    const values=p[key]??[];
    if(!Array.isArray(values)||values.length>20||values.some(x=>typeof x!=='string'||!known.has(x)))throw new Invalid('Select valid catalog '+key+'.');
    p[key]=[...new Set(values)];
  }
  if(p.budget!=null&&(typeof p.budget!=='number'||!Number.isFinite(p.budget)||p.budget<=0||p.budget>10000))throw new Invalid('Budget must be a positive number up to 10000.');
  for(const key of ['open_after_8pm','exclude_seen'])if(key in p&&typeof p[key]!=='boolean')throw new Invalid(key+' must be boolean.');
  return p;
}
export function recommend(preferences={},model='baseline',k=5){
  keys(preferences,['cuisines','budget','open_after_8pm','seed_item_ids','exclude_seen']);
  if(!['baseline','two_tower'].includes(model))throw new Invalid('Choose baseline or two_tower.');
  if(!Number.isInteger(k)||k<1||k>20)throw new Invalid('k must be 1–20.');
  const p=validatePreferences({...preferences}),dim=DATA.features[0].length,q=Array(dim).fill(0);
  let weight=0;
  for(const id of p.seed_item_ids){const v=DATA.features[itemIndex.get(id)];v.forEach((x,j)=>q[j]+=2*x);weight+=2;}
  if(p.cuisines.length){const matches=DATA.items.map((item,i)=>p.cuisines.includes(item.cuisine)?i:-1).filter(i=>i>=0);if(matches.length){for(const i of matches)DATA.features[i].forEach((x,j)=>q[j]+=2*x/matches.length);weight+=2;}}
  const query=norm(q.map(x=>weight?x/weight:0)),context=query.some(x=>x!==0);
  const embedding=DATA.query_weight.map(w=>Math.tanh(dot(w,[...query,1])));
  const retrieval=DATA.items.map((item,i)=>model==='two_tower'?dot(DATA.item_embeddings[i],embedding)/.2+DATA.item_bias[i]:context?dot(DATA.features[i],query)+1e-4*(DATA.popularity[i]-3):DATA.popularity[i]);
  const r=minmax(retrieval),pop=minmax(DATA.popularity),scores=r.map((x,i)=>.85*x+.15*pop[i]);
  const eligible=DATA.items.map((item,i)=>({item,i})).filter(({item})=>(!p.cuisines.length||p.cuisines.includes(item.cuisine))&&(p.budget==null||(item.average_cost!=null&&item.average_cost<=p.budget))&&(!p.open_after_8pm||item.open_after_8pm===true)&&(!p.exclude_seen||!p.seed_item_ids.includes(item.item_id)));
  eligible.sort((a,b)=>scores[b.i]-scores[a.i]||a.item.item_id.localeCompare(b.item.item_id));
  const rows=eligible.slice(0,k).map(({item,i})=>{
    const reasons=[];if(p.cuisines.length)reasons.push('Matches your selected cuisine.');if(p.budget!=null)reasons.push('Historical average cost fits your budget.');if(p.open_after_8pm)reasons.push('Historical catalog marks this restaurant open after 8 pm.');reasons.push(context?'Retrieved using your selected restaurant and cuisine preferences.':'Global prior: no selected metadata preference is available.');if(item.metadata_status==='review_only')reasons.push('Review-only restaurant: catalog details are unknown.');if(model==='two_tower')reasons.push('Two Tower similarity is an uncalibrated retrieval score.');
    const round=x=>Math.round(x*1e6)/1e6;
    return{...item,reasons,score:round(scores[i]),ranking_score:round(scores[i]),retrieval_score:round(retrieval[i])};
  });
  return{recommendations:rows,request_id:crypto.randomUUID(),model,requested_model:model,model_version:model==='two_tower'?DATA.model_version:'baseline_v1',candidate_count:eligible.length,limitations:DATA.limitations,ranking_policy:DATA.ranking_policy,message:eligible.length?null:'No restaurants meet every selected constraint.'};
}
function response(body,status=200,type='application/json; charset=utf-8'){
  return new Response(type.startsWith('text/html')?body:JSON.stringify(body),{status,headers:{'Content-Type':type,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'"}});
}
async function body(request){
  const origin=request.headers.get('Origin');if(origin&&origin!==new URL(request.url).origin)throw new Invalid('Cross-origin writes are disabled.');
  if(request.headers.get('Content-Type')?.split(';')[0]!=='application/json')throw new Invalid('Use application/json.');
  if(Number(request.headers.get('Content-Length'))>16384)throw new Invalid('Request exceeds 16 KiB.');
  const reader=request.body?.getReader();if(!reader)throw new Invalid('Provide a JSON object.');let length=0,chunks=[];
  while(true){const{value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>16384){await reader.cancel();throw new Invalid('Request exceeds 16 KiB.');}chunks.push(value);}
  const bytes=new Uint8Array(length);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
  let value;try{value=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));}catch{throw new Invalid('Provide valid JSON.');}return value;
}
function database(env){if(!env.DB)throw Error('Feedback store unavailable.');return env.DB;}
async function counts(db){const since=Date.now()-WEEK;const impressions=await db.prepare('SELECT COUNT(*) AS n FROM impressions WHERE timestamp>=?').bind(since).first('n');const events=await db.prepare('SELECT type,COUNT(*) AS n FROM events WHERE timestamp>=? GROUP BY type').bind(since).all();const n=Object.fromEntries(events.results.map(row=>[row.type,row.n]));return{impressions,clicks:n.click??0,favorites:n.favorite??0};}
async function maintain(db){await db.batch([db.prepare('DELETE FROM events WHERE timestamp<?').bind(Date.now()-WEEK),db.prepare('DELETE FROM impressions WHERE timestamp<?').bind(Date.now()-WEEK)]);}
export default{async fetch(request,env){
  const path=new URL(request.url).pathname;
  try{
    if(request.method==='GET'){
      if(path==='/')return response(PAGE,200,'text/html; charset=utf-8');
      if(path==='/api/catalog')return response(DATA.items);
      if(path==='/api/metrics')return response(REPORT);
      if(path==='/api/health')return response({status:'ok',service:'public_portfolio_demo',synthetic:true,log_source:'portfolio_demo',log_counts:await counts(database(env))});
      return response({error:'Route not found.'},404);
    }
    if(request.method!=='POST'||!['/api/recommend','/api/events'].includes(path))return response({error:'Route not found.'},404);
    const input=await body(request),db=database(env);
    keys(input,path==='/api/recommend'?['session_id','consent','preferences','model','k']:['session_id','consent','request_id','item_id','type']);
    const sid=session(input.session_id);if(input.consent!==true)throw new Invalid('Accept anonymous demo logging before submitting.');
    if(path==='/api/recommend'){
      const result=recommend(input.preferences,input.model??'baseline',input.k??5);
      await maintain(db);
      const recent=await db.prepare('SELECT COUNT(DISTINCT request_id) AS n FROM impressions WHERE session_id=? AND timestamp>?').bind(sid,Date.now()-60000).first('n');if(recent>=60)return response({error:'Please wait before generating more recommendations.'},429);
      const now=Date.now();if(result.recommendations.length)await db.batch(result.recommendations.map((item,i)=>db.prepare('INSERT INTO impressions (id,request_id,session_id,item_id,position,model,model_version,timestamp,synthetic,source) VALUES (?,?,?,?,?,?,?,?,1,\'portfolio_demo\')').bind(crypto.randomUUID(),result.request_id,sid,item.item_id,i+1,result.model,result.model_version,now)));
      result.logging={consent:true,synthetic:true,source:'portfolio_demo',evaluation_input:false,impressions_recorded:result.recommendations.length};return response(result);
    }
    if(!['click','favorite'].includes(input.type))throw new Invalid('Choose click or favorite.');
    if(typeof input.request_id!=='string'||input.request_id.length>80||typeof input.item_id!=='string'||input.item_id.length>128)throw new Invalid('Use a previously displayed item and request.');
    const row=await db.prepare('SELECT * FROM impressions WHERE request_id=? AND session_id=? AND item_id=? AND timestamp>=?').bind(input.request_id,sid,input.item_id,Date.now()-WEEK).first();if(!row)throw new Invalid('No matching impression for this browser, request and item.');
    const id=crypto.randomUUID();await db.prepare('INSERT INTO events (id,impression_id,request_id,session_id,item_id,type,position,model,model_version,timestamp,synthetic,source) VALUES (?,?,?,?,?,?,?,?,?,?,1,\'portfolio_demo\')').bind(id,row.id,row.request_id,row.session_id,row.item_id,input.type,row.position,row.model,row.model_version,Date.now()).run();return response({ok:true,event_id:id,type:input.type,synthetic:true,source:'portfolio_demo',evaluation_input:false},201);
  }catch(error){if(error instanceof Invalid)return response({error:error.message},400);return response({error:'The recommendation or feedback service is temporarily unavailable. Your preferences remain on this page; please retry.'},503);}
}};

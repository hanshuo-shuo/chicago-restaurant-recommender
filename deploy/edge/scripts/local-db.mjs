import {DatabaseSync} from 'node:sqlite';
import {readdirSync,readFileSync} from 'node:fs';
export function localDB(filename=':memory:'){
  const db=new DatabaseSync(filename);db.exec('PRAGMA foreign_keys=ON');
  db.exec('CREATE TABLE IF NOT EXISTS _preview_migrations (name TEXT PRIMARY KEY)');
  for(const name of readdirSync('drizzle').filter(n=>n.endsWith('.sql')).sort()){
    if(!db.prepare('SELECT name FROM _preview_migrations WHERE name=?').get(name)){
      db.exec(readFileSync('drizzle/'+name,'utf8'));db.prepare('INSERT INTO _preview_migrations VALUES (?)').run(name);
    }
  }
  const prepared=(sql,values=[])=>({
    bind(...args){return prepared(sql,args);},
    async first(column){const row=db.prepare(sql).get(...values);return column?row?.[column]??null:row??null;},
    async all(){return{results:db.prepare(sql).all(...values)};},
    async run(){return db.prepare(sql).run(...values);},
    execute(){return db.prepare(sql).run(...values);},
  });
  return{prepare:prepared,async batch(statements){db.exec('BEGIN');try{const results=statements.map(s=>s.execute());db.exec('COMMIT');return results;}catch(e){db.exec('ROLLBACK');throw e;}},close(){db.close();}};
}

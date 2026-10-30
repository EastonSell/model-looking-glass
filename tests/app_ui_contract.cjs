'use strict';
const fs=require('fs'), vm=require('vm'), assert=require('assert');
// Minimal DOM/API contract test; this is NOT a rendering or real-browser test.
const root=require('path').join(__dirname,'..','looking_glass','app_static')+require('path').sep;
const html=fs.readFileSync(root+'index.html','utf8');
class Element {
 constructor(tag='div',id=''){this.tag=tag;this.id=id;this.children=[];this.events={};this.value='';this.hidden=false;this.disabled=false;this.checked=false;this.attributes={};this._text='';this.classList={toggle:()=>{}};this.labels=[{textContent:id}];}
 set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(x=>x.textContent??'').join('');}
 append(...items){this.children.push(...items);} replaceChildren(...items){this._text='';this.children=[...items];}
 addEventListener(name,fn){(this.events[name]??=[]).push(fn);} setAttribute(k,v){this.attributes[k]=v;}
 async fire(name){for(const fn of this.events[name]??[])await fn();}
}
const ids={};for(const m of html.matchAll(/<([\w-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){let e=ids[m[3]]=new Element(m[1],m[3]);for(const key of ['value','min','max','type']){const a=m[2].match(new RegExp(key+'="([^"]*)"'));if(a)e[key]=a[1];}e.hidden=/\bhidden\b/.test(m[2]);e.disabled=/\bdisabled\b/.test(m[2]);}
ids['capture-mode'].value='auto';ids['vector-view'].value='raw';
function descendants(e,tag){return e.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...descendants(c,tag)]:[]);}
const document={getElementById:id=>{assert(ids[id],id);return ids[id]},querySelector:s=>s.startsWith('meta')?{content:'unit-test-token'}:new Element('label'),querySelectorAll:s=>s.includes('target-options')?descendants(ids['target-options'],'input').filter(e=>!s.includes(':checked')||e.checked):s.includes('event-body')?descendants(ids['event-body'],'button'):[],createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>({textContent:text})};
const info={family:'gpt2',name:'Tiny local model',targets:['attn','mlp','block'],layers:2};
const run={answer:'<img src=x onerror=bad()>',task:'greedy',event_count:1,total_bytes:48,coverage:{complete:true},steps:[{forward_index:0}]};
const event={index:0,metadata:{module_path:'transformer.h.0.attn',forward_index:0},shape:[1,3,4],dtype:'torch.float32',tensor_bytes:48,vector_count:3,vector_width:4};
const calls=[];let failTensor=false;
async function fetch(url,options){assert.equal(options.headers['X-MLG-Token'],'unit-test-token');assert(url.startsWith('/api/'));calls.push([url,options]);let data;const path=url.slice(5);if(path==='status')data={model:null,run:null,models:[]};else if(path==='inspect'||path==='load')data=info;else if(path==='run'||path==='open')data=run;else if(path.startsWith('events?'))data={events:[event],total:1};else if(path.startsWith('tensor?'))data=failTensor?{error:'Injected tensor read error'}:{vector_width:4,offset:0,values:[0,-0,1.5,-2.25],raw_bits:['0x0','0x80000000','0x3fc00000','0xc0100000'],bins:[{start:0,end:2,mean:0,min:0,max:0},{start:2,end:4,mean:-.375,min:-2.25,max:1.5}]};else if(path==='save')data={path:'exports/my-capture'};else if(path==='clear')data={};else throw Error(path);return {ok:!data.error,status:data.error?400:200,json:async()=>data};}
vm.runInNewContext(fs.readFileSync(root+'app.js','utf8'),{document,fetch,console,FileReader:class{},setTimeout,clearTimeout});
(async()=>{
 await new Promise(setImmediate);assert(ids['run-button'].disabled);assert(!ids['load-button'].disabled);
 ids['model-path'].value='/local/checkpoint';await ids['inspect-button'].fire('click');assert(ids['model-info'].textContent.startsWith('Inspection only'));
 await ids['load-button'].fire('click');assert(!ids['run-button'].disabled);
 ids.prompt.value='Hello';await ids['run-button'].fire('click');assert(!ids['result-panel'].hidden);assert.equal(ids.answer.textContent,run.answer);assert.equal(ids['value-body'].children.length,4);assert.equal(ids['value-body'].children[1].children[1].textContent,'-0');assert(!descendants(ids['event-body'],'button')[0].disabled);
 ids['vector-view'].value='reduced';await ids['vector-view'].fire('change');assert(!ids['reduced-values'].hidden);assert.equal(ids['bin-body'].children.length,2);assert.equal(descendants(ids['bin-plot'],'rect').length,2);
 await ids['save-button'].fire('click');assert(ids['save-status'].textContent.startsWith('Saved explicitly'));
 failTensor=true;await descendants(ids['event-body'],'button')[0].fire('click');assert.equal(ids.notice.textContent,'Injected tensor read error');failTensor=false;
 await ids['clear-button'].fire('click');assert(ids['result-panel'].hidden);assert.equal(ids.prompt.value,'');
 ids['open-path'].value='/local/saved';await ids['open-button'].fire('click');assert(!ids['result-panel'].hidden);
 ids['capture-mode'].value='custom';await ids['capture-mode'].fire('change');assert(!ids['custom-capture'].hidden);ids.layers.value='0,0';await ids['run-button'].fire('click');assert(ids.notice.textContent.includes('distinct'));
 const post=calls.find(c=>c[0]==='/api/run');assert.equal(JSON.parse(post[1].body).settings.targets,'auto');assert.equal(JSON.parse(post[1].body).settings.max_trace_bytes,134217728);
 console.log(JSON.stringify({dom_contract_checks:'passed',authenticated_api_requests:calls.length,tests:['initial state','inspect/load','run/full raw values','signed zero','buttons reenabled','approximate bins','explicit export','tensor error visible','clear','open saved trace','custom layer validation']}));
})().catch(e=>{console.error(e);process.exitCode=1});

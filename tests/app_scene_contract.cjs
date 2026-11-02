'use strict';
// Real Three geometry/camera/raycast math. Renderer boundary is fake: not GPU/browser proof.
const assert=require('assert'),fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.join(__dirname,'../looking_glass/app_static');
assert(fs.existsSync(path.join(root,'app_scene.js')),'scene renderer module must exist');
const THREE=require(path.join(root,'vendor/three-0.128.0.min.js'));
class El {constructor(){this.style={};this.children=[];this.listeners={};this.clientWidth=1600;this.clientHeight=900;} appendChild(e){this.children.push(e);e.parentNode=this;} remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(x=>x!==this);} setAttribute(){} addEventListener(k,f){this.listeners[k]=f;} removeEventListener(k){delete this.listeners[k];} getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth,height:this.clientHeight};}}
let reduced=true,renderFailure=false,constructFailure=false;
let renders=[],disposed=0,raf=new Map(),serial=0;
class Renderer {constructor(){if(constructFailure)throw Error('GPU unavailable');this.domElement=new El();this.domElement.ownerDocument=document;}setPixelRatio(){}setSize(){}render(scene,camera){if(renderFailure)throw Error('GPU lost');renders.push({scene,camera});}dispose(){disposed++;}}
const document=new El();document.createElement=()=>new El();document.hidden=false;
const context={THREE:{...THREE,WebGLRenderer:Renderer},document,console,performance:{now:()=>0},matchMedia:()=>({matches:reduced,addEventListener(){},removeEventListener(){}}),requestAnimationFrame:f=>{raf.set(++serial,f);return serial;},cancelAnimationFrame:id=>raf.delete(id),devicePixelRatio:1,getComputedStyle:()=>({getPropertyValue:n=>({'--accent':'#112233','--muted':'#445566','--text-3':'#778899'}[n]||'')})};context.window=context;
vm.runInNewContext(fs.readFileSync(path.join(root,'vendor/OrbitControls-0.128.0.js'),'utf8'),context);
vm.runInNewContext(fs.readFileSync(path.join(root,'app_scene.js'),'utf8'),context);
let viewInsets={left:360,right:560};
const el=new El(),selected=[],fail=[];const api=context.MLGScene.create(el,{getViewInsets:()=>viewInsets,onSelect:k=>selected.push(k),onUnavailable:r=>fail.push(r)});
assert(api);api.setLayers([{key:'L0',label:'Layer 0',sites:[{key:'a',label:'Attention'},{key:'b',label:'MLP'}]},{key:'L1',label:'Layer 1',sites:[{key:'c',label:'Block'}]}]);
function frame(){const calls=[...raf.values()];raf.clear();calls.forEach(f=>f(arguments[0]||0));}frame();
let {scene,camera}=renders.at(-1),objects=[];scene.traverse(o=>objects.push(o));
assert.equal(camera.fov,35);assert(camera.isPerspectiveCamera);
const planes=objects.filter(o=>o.userData.kind==='layer');assert.equal(planes.length,2);assert.equal(planes[0].material.color.getHexString(),'112233','plane uses CSS accent token');assert.equal(planes[0].geometry.parameters.width,1);assert.equal(planes[0].geometry.parameters.height,1.3);assert.equal(planes[0].material.depthWrite,false);assert.equal(planes[0].material.side,THREE.DoubleSide);
const sites=objects.filter(o=>o.userData.siteKey);assert.deepEqual(sites.map(o=>o.userData.siteKey),['a','b','c']);assert(!objects.some(o=>o.isPoints),'no fabricated neuron dots');assert.equal(objects.filter(o=>o.isLineSegments).length,planes.length+sites.length,'only portrait and module rectangle edges, no invented connections');
// Every portrait plane fits the available 360..1040 desktop gap, not behind either dock.
camera.updateMatrixWorld();scene.updateMatrixWorld(true);
for(const plane of planes)for(const x of [-0.5,0.5])for(const y of [-0.65,0.65]){
 const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera);
 const screenX=(p.x+1)*800,screenY=(1-p.y)*450;
 assert(screenX>=360 && screenX<=1040,`plane projected outside desktop gap: ${screenX}`);
 assert(screenY>=0 && screenY<=900);
}
for(const viewport of [1440,2576]){
 el.clientWidth=viewport;viewInsets={left:Math.max(300,viewport*.225),right:Math.max(400,viewport*.35)};api.resize();frame();camera.updateMatrixWorld();
 const left=Math.max(300,viewport*.225),right=viewport-Math.max(400,viewport*.35);
 for(const plane of planes)for(const x of [-.5,.5])for(const y of [-.65,.65]){
  const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera),sx=(p.x+1)*viewport/2;
  assert(sx>=left && sx<=right,`plane outside ${viewport}px dock gap`);
 }
}
el.clientWidth=1200;viewInsets={left:0,right:0};api.resize();frame();camera.updateMatrixWorld();const centered=new THREE.Box3().setFromObject(scene).getCenter(new THREE.Vector3()).project(camera);assert(Math.abs(centered.x)<1e-8,'stacked responsive layout under1280 uses full-width framing');
el.clientWidth=1600;viewInsets={left:360,right:560};api.resize();frame();
// Collapsing either side must reclaim real camera space, not merely hide its DOM.
for(const width of [1280,1440,1920]){
 el.clientWidth=width;
 const fullLeft=Math.max(300,width*.225),fullRight=Math.max(400,width*.35);
 const spans=[];
 for(const insets of [{left:fullLeft,right:fullRight},{left:50,right:fullRight},{left:fullLeft,right:50},{left:50,right:50}]){
  viewInsets=insets;api.resize();frame();camera.updateMatrixWorld();
  const xs=[];
  for(const plane of planes)for(const x of [-.5,.5])for(const y of [-.65,.65]){
   const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera),sx=(p.x+1)*width/2;
   assert(sx>=insets.left && sx<=width-insets.right,'scene stays outside visible dock/rail');xs.push(sx);
  }
  spans.push(Math.max(...xs)-Math.min(...xs));
  const center=new THREE.Box3().setFromObject(scene).getCenter(new THREE.Vector3()).project(camera);
  assert(Math.abs((center.x+1)*width/2-(insets.left+(width-insets.left-insets.right)/2))<1e-8,'camera recenters in the actual available gap');
 }
 for(const span of spans.slice(1))assert(span>=spans[0]-1e-8,'collapse never shrinks layers');
 if(width<1920){assert(spans[1]>spans[0]*1.1,'left collapse makes width-constrained layers larger');assert(spans[2]>spans[0]*1.1,'right collapse makes width-constrained layers larger');}
 assert(spans[3]>=Math.max(spans[1],spans[2])-1e-8,'both collapsed never shrink the view; height may already limit scale');
 camera.updateMatrixWorld();const center=new THREE.Box3().setFromObject(scene).getCenter(new THREE.Vector3()).project(camera);
 assert(Math.abs(center.x)<1e-8,'equal collapsed rails center the layer stack');
}
for(const invalid of [undefined,{left:NaN,right:-5},{left:Infinity,right:Infinity},{left:900,right:900}]){
 viewInsets=invalid;el.clientWidth=800;api.resize();frame();
 assert(Number.isFinite(camera.position.length()),'invalid or stale geometry never produces an invalid camera');
}
el.clientWidth=1600;viewInsets={left:360,right:560};api.resize();frame();
const layerLabel=el.children[1].children[0],siteLabel=el.children[1].children[1];
const labelCorner=new THREE.Vector3(-.5,.65,0).applyMatrix4(planes[0].matrixWorld).project(camera);
assert(Math.abs(parseFloat(layerLabel.style.left)-(labelCorner.x+1)*800)<1e-6,'layer label anchors at actual top-left plane corner');
assert(Math.abs(parseFloat(layerLabel.style.top)-(1-labelCorner.y)*450)<1e-6);
assert(parseFloat(layerLabel.style.left)+6>=360 && parseFloat(layerLabel.style.left)+6<1040,'layer label anchor remains in dock gap');assert(parseFloat(layerLabel.style.top)+6>=0 && parseFloat(layerLabel.style.top)+6<900,'layer label anchor remains in scene height');
assert.equal(layerLabel.style.transform,'translate(6px, 6px)');assert.equal(siteLabel.style.transform,'translate(-50%, -50%)','module labels remain centered on pick targets');
assert.equal(layerLabel.style.color,'var(--text-3)');assert(!layerLabel.style.textShadow,'no label glow/shadow');
const clickCanvas=el.children.find(e=>e.listeners.webglcontextlost);
function pointFor(object){camera.updateMatrixWorld();scene.updateMatrixWorld(true);const p=object.getWorldPosition(new THREE.Vector3()).project(camera);return {clientX:(p.x+1)*800,clientY:(1-p.y)*450};}
const click=pointFor(sites[0]);clickCanvas.listeners.pointerdown({...click,button:2});clickCanvas.listeners.pointerup({...click,button:2});assert.equal(selected.length,0,'secondary button must not select');
clickCanvas.listeners.pointerdown({...click,pointerId:1});clickCanvas.listeners.pointerup({...click,pointerId:2});assert.equal(selected.length,0,'different pointer must not select');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointercancel();clickCanvas.listeners.pointerup(click);assert.equal(selected.length,0,'cancelled pointer must not select');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointerup(click);
assert.deepEqual(selected,['a'],'raycast returns exact captured site key');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointerup({...click,clientX:click.clientX+30});assert.equal(selected.length,1,'drag is not selection');
let released=0;for(const o of objects){if(o.geometry)o.geometry.addEventListener('dispose',()=>released++);if(o.material)o.material.addEventListener('dispose',()=>released++);}
assert.equal(api.selectSite('a'),true);frame();frame();assert.equal(api.selectSite('a'),true);assert.equal(raf.size,0,'same-key UI synchronization must not restart focus');assert.equal(api.selectSite('missing'),false);
const selectedOpacity=sites[0].material.opacity,sceneObjects=scene.children.slice();camera.position.set(80,40,-20);camera.lookAt(new THREE.Vector3(0,0,0));api.resize();frame();camera.updateMatrixWorld();assert.deepEqual(scene.children,sceneObjects);assert.equal(sites[0].material.opacity,selectedOpacity,'reset framing preserves selected-site highlight');for(const plane of planes)for(const x of [-.5,.5])for(const y of [-.65,.65]){const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera);assert((p.x+1)*800>=360&&(p.x+1)*800<=1040);assert((1-p.y)*450>=0&&(1-p.y)*450<=900);}assert.equal(selected.length,1,'reset does not select or read a site');
assert.throws(()=>api.setLayers([{key:'x',sites:[{key:'same'},{key:'same'}]}]),/unique/i);
assert.throws(()=>api.setLayers(Array.from({length:13},(_,i)=>({key:String(i),sites:[]}))),/12/);
clickCanvas.listeners.pointerdown(click);
api.setLayers([{key:'new',label:'Replacement',sites:[{key:'fresh',label:'Fresh'}]}]);assert.equal(api.selectSite('a'),false);clickCanvas.listeners.pointerup(click);assert.equal(selected.length,1,'replacement invalidates in-flight pointer');assert(released>0,'replaced geometry and material disposed');frame();
assert.equal(raf.size,0,'reduced motion settles without an idle render loop');
api.setActive(false);assert.equal(raf.size,0);api.setActive(true);frame();api.resize();
const canvas=el.children.find(e=>e.listeners.webglcontextlost);assert(canvas);canvas.listeners.webglcontextlost({preventDefault(){}});assert.equal(fail.length,1);assert.equal(raf.size,0);assert.equal(api.selectSite('fresh'),false);
api.dispose();api.dispose();assert.equal(disposed,1);assert.equal(el.children.length,0);
// Animated focus uses real OrbitControls and a 600 ms transition when motion is permitted.
reduced=false;
const movingEl=new El(),moving=context.MLGScene.create(movingEl,{});
const fullPage=Array.from({length:12},(_,i)=>({key:'layer'+i,sites:Array.from({length:6},(_,j)=>({key:`s${i}-${j}`}))}));
assert.doesNotThrow(()=>moving.setLayers(fullPage));frame();let bounded=[];renders.at(-1).scene.traverse(o=>bounded.push(o));assert.equal(bounded.filter(o=>o.userData.siteKey).length,72);assert.equal(bounded.filter(o=>o.userData.kind==='layer').length,12);assert.equal(bounded.filter(o=>o.geometry).length,168,'fixed resource bound: two geometries per layer and site');
fullPage[0].sites.push({key:'site73'});assert.throws(()=>moving.setLayers(fullPage),/72/);
moving.setLayers([{key:'animated',label:'L'.repeat(200),sites:[{key:'focus',label:'M'.repeat(200)},{key:'other'}]}]);assert(movingEl.children[1].children.every(e=>e.textContent.length<=64),'visual labels must remain bounded');frame();
const movingCamera=renders.at(-1).camera,before=movingCamera.position.clone();
moving.selectSite('focus');frame(300);assert(raf.size>0,'animation schedules continuation');frame(600);frame(601);assert(movingCamera.position.distanceTo(before)>0,'focus actually moves camera');
moving.selectSite('other');frame(200);moving.selectSite(null);const cancelled=movingCamera.position.clone();frame(600);assert(movingCamera.position.distanceTo(cancelled)<1e-8,'clearing selection cancels old focus animation');
// Hidden documents pause work; visibility resumes a single requested update.
document.hidden=true;document.listeners.visibilitychange();assert.equal(raf.size,0);document.hidden=false;document.listeners.visibilitychange();assert(raf.size>0);frame(602);
moving.dispose();assert.equal(movingEl.children.length,0);
constructFailure=true;const missingEl=new El(),missingReasons=[];
const missing=context.MLGScene.create(missingEl,{onUnavailable:r=>missingReasons.push(r)});assert.equal(missingReasons.length,1);assert.equal(missing.selectSite('x'),false);missing.dispose();assert.equal(missingEl.children.length,0);constructFailure=false;
const brokenEl=new El(),brokenReasons=[],broken=context.MLGScene.create(brokenEl,{onUnavailable:r=>brokenReasons.push(r)});renderFailure=true;frame();assert.equal(brokenReasons.length,1);assert.equal(raf.size,0);broken.dispose();renderFailure=false;
console.log('app_scene_contract: PASS (real Three math; fake renderer, not GPU proof)');

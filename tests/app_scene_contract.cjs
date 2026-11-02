'use strict';
// Real Three geometry/camera/raycast math. Renderer boundary is fake: not GPU/browser proof.
const assert=require('assert'),fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.join(__dirname,'../looking_glass/app_static');
assert(fs.existsSync(path.join(root,'app_scene.js')),'scene renderer module must exist');
const THREE=require(path.join(root,'vendor/three-0.128.0.min.js'));
class El {constructor(){this.style={};this.children=[];this.listeners={};this.attributes={};this.clientWidth=1600;this.clientHeight=900;} appendChild(e){this.children.push(e);e.parentNode=this;} remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(x=>x!==this);} setAttribute(k,v){this.attributes[k]=v;} addEventListener(k,f){this.listeners[k]=f;} removeEventListener(k){delete this.listeners[k];} getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth,height:this.clientHeight};}}
let reduced=true,renderFailure=false,constructFailure=false;
let renders=[],disposed=0,raf=new Map(),serial=0,now=0;
class Renderer {constructor(){if(constructFailure)throw Error('GPU unavailable');this.domElement=new El();this.domElement.ownerDocument=document;}setPixelRatio(){}setSize(){}render(scene,camera){if(renderFailure)throw Error('GPU lost');renders.push({scene,camera});}dispose(){disposed++;}}
const document=new El();document.createElement=()=>new El();document.hidden=false;
const context={THREE:{...THREE,WebGLRenderer:Renderer},document,console,performance:{now:()=>now},matchMedia:()=>({matches:reduced,addEventListener(){},removeEventListener(){}}),requestAnimationFrame:f=>{raf.set(++serial,f);return serial;},cancelAnimationFrame:id=>raf.delete(id),devicePixelRatio:1,getComputedStyle:()=>({getPropertyValue:n=>({'--accent':'#112233','--selected':'#eeddaa','--muted':'#445566','--text-3':'#778899'}[n]||'')})};context.window=context;
vm.runInNewContext(fs.readFileSync(path.join(root,'vendor/OrbitControls-0.128.0.js'),'utf8'),context);
const OriginalControls=context.THREE.OrbitControls;let latestControls;context.THREE.OrbitControls=function(...args){return latestControls=new OriginalControls(...args);};
vm.runInNewContext(fs.readFileSync(path.join(root,'app_scene.js'),'utf8'),context);
let viewInsets={left:360,right:560};
const el=new El(),selected=[],fail=[];const api=context.MLGScene.create(el,{getViewInsets:()=>viewInsets,onSelect:k=>selected.push(k),onUnavailable:r=>fail.push(r)});
assert(api);api.setLayers([{key:'L0',label:'Layer 0',sites:[{key:'a',label:'Attention'},{key:'b',label:'MLP'}]},{key:'L1',label:'Layer 1',sites:[{key:'c',label:'Block'}]}]);
function frame(){const calls=[...raf.values()];raf.clear();calls.forEach(f=>f(arguments[0]||0));}frame();
let {scene,camera}=renders.at(-1),objects=[];scene.traverse(o=>objects.push(o));
assert.equal(camera.fov,35);assert(camera.isPerspectiveCamera);
const planes=objects.filter(o=>o.userData.kind==='layer');assert.equal(planes.length,2);assert.equal(planes[0].material.color.getHexString(),'112233','plane uses CSS accent token');assert.equal(planes[0].geometry.parameters.width,0.9);assert.equal(planes[0].geometry.parameters.height,1.65);assert.equal(planes[0].material.depthWrite,false);assert.equal(planes[0].material.side,THREE.DoubleSide);
const sites=objects.filter(o=>o.userData.siteKey);assert.deepEqual(sites.map(o=>o.userData.siteKey),['a','b','c']);assert(!objects.some(o=>o.isPoints),'no fabricated neuron dots');assert.equal(objects.filter(o=>o.isLineSegments).length,planes.length+sites.length,'only layer and invisible site outlines, no invented connections');assert(objects.filter(o=>o.isLineSegments).filter(o=>o.material.opacity===0).length===sites.length,'site outlines are invisible');assert(sites.every(o=>o.material.opacity===0),'hit targets are invisible');
const markers=objects.filter(o=>o.geometry && o.geometry.type==='CircleGeometry');assert.equal(markers.length,sites.length,'one marker per captured site');
for(const site of sites) assert.equal(markers.filter(m=>m.position.x===site.position.x && m.position.y===site.position.y).length,1,'each hit target has exactly one aligned site marker');
// Every portrait plane fits the available 360..1040 desktop gap, not behind either dock.
camera.updateMatrixWorld();scene.updateMatrixWorld(true);
for(const plane of planes)for(const x of [-0.45,0.45])for(const y of [-0.825,0.825]){
 const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera);
 const screenX=(p.x+1)*800,screenY=(1-p.y)*450;
 assert(screenX>=360 && screenX<=1040,`plane projected outside desktop gap: ${screenX}`);
 assert(screenY>=0 && screenY<=900);
}
for(const viewport of [1440,2576]){
 el.clientWidth=viewport;viewInsets={left:Math.max(300,viewport*.225),right:Math.max(400,viewport*.35)};api.resize();frame();camera.updateMatrixWorld();
 const left=Math.max(300,viewport*.225),right=viewport-Math.max(400,viewport*.35);
 for(const plane of planes)for(const x of [-.45,.45])for(const y of [-.825,.825]){
  const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera),sx=(p.x+1)*viewport/2;
  assert(sx>=left && sx<=right,`plane outside ${viewport}px dock gap`);
 }
}
el.clientWidth=1200;viewInsets={left:0,right:0};api.resize();frame();camera.updateMatrixWorld();const centered=planes[0].getWorldPosition(new THREE.Vector3()).project(camera);assert(Math.abs(centered.x)<1e-8,'stacked responsive layout under1280 uses full-width framing');
el.clientWidth=1600;viewInsets={left:360,right:560};api.resize();frame();
// Collapsing either side must reclaim real camera space, not merely hide its DOM.
for(const width of [1280,1440,1920]){
 el.clientWidth=width;
 const fullLeft=Math.max(300,width*.225),fullRight=Math.max(400,width*.35);
 const spans=[];
 for(const insets of [{left:fullLeft,right:fullRight},{left:50,right:fullRight},{left:fullLeft,right:50},{left:50,right:50}]){
  viewInsets=insets;api.resize();frame();camera.updateMatrixWorld();
  const xs=[];
  for(const plane of planes)for(const x of [-.45,.45])for(const y of [-.825,.825]){
   const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera),sx=(p.x+1)*width/2;
   assert(sx>=insets.left && sx<=width-insets.right,'scene stays outside visible dock/rail');xs.push(sx);
  }
  spans.push(Math.max(...xs)-Math.min(...xs));
  const center=planes[0].getWorldPosition(new THREE.Vector3()).project(camera);
  assert(Math.abs((center.x+1)*width/2-(insets.left+(width-insets.left-insets.right)/2))<1e-8,'camera recenters in the actual available gap');
 }
 for(const span of spans.slice(1))assert(span>=spans[0]-1e-8,'collapse never shrinks layers');
 if(width<1920){assert(spans[1]>spans[0]*1.1,'left collapse makes width-constrained layers larger');assert(spans[2]>spans[0]*1.1,'right collapse makes width-constrained layers larger');}
 assert(spans[3]>=Math.max(spans[1],spans[2])-1e-8,'both collapsed never shrink the view; height may already limit scale');
 camera.updateMatrixWorld();const center=planes[0].getWorldPosition(new THREE.Vector3()).project(camera);
 assert(Math.abs(center.x)<1e-8,'equal collapsed rails center the layer stack');
}
for(const invalid of [undefined,{left:NaN,right:-5},{left:Infinity,right:Infinity},{left:900,right:900}]){
 viewInsets=invalid;el.clientWidth=800;api.resize();frame();
 assert(Number.isFinite(camera.position.length()),'invalid or stale geometry never produces an invalid camera');
}
el.clientWidth=1600;viewInsets={left:360,right:560};api.resize();frame();
const layerLabel=el.children[1].children[0],siteLabel=el.children[1].children[1];
const labelCorner=new THREE.Vector3(-.45,.825,0).applyMatrix4(planes[0].matrixWorld).project(camera);
assert(Math.abs(parseFloat(layerLabel.style.left)-(labelCorner.x+1)*800)<1e-6,'layer label anchors at actual top-left plane corner');
assert(Math.abs(parseFloat(layerLabel.style.top)-(1-labelCorner.y)*450)<1e-6);
assert(parseFloat(layerLabel.style.left)+6>=360 && parseFloat(layerLabel.style.left)+6<1040,'layer label anchor remains in dock gap');assert(parseFloat(layerLabel.style.top)+6>=0 && parseFloat(layerLabel.style.top)+6<900,'layer label anchor remains in scene height');
assert.equal(layerLabel.style.transform,'translate(6px, 6px)');assert.equal(siteLabel.style.transform,'translate(8px, -50%)','module labels sit beside pick targets');
assert.equal(siteLabel.style.display,'none','unselected module text must not crowd the wireframe');
assert.equal(layerLabel.style.color,'var(--text-3)');assert(!layerLabel.style.textShadow,'no label glow/shadow');
const clickCanvas=el.children.find(e=>e.listeners.webglcontextlost);
function pointFor(object){camera.updateMatrixWorld();scene.updateMatrixWorld(true);const p=object.getWorldPosition(new THREE.Vector3()).project(camera);return {clientX:(p.x+1)*800,clientY:(1-p.y)*450};}
// Hover discovers real captured sites without selection, camera motion, or an idle loop.
assert.equal(typeof clickCanvas.listeners.pointermove,'function','captured-site hover must be discoverable');
assert.equal(typeof clickCanvas.listeners.pointerleave,'function');
const hoverPoint=pointFor(sites[0]),hoverCamera=camera.position.clone();
let hoverRays=0;const originalIntersect=THREE.Raycaster.prototype.intersectObjects;
THREE.Raycaster.prototype.intersectObjects=function(...args){hoverRays++;return originalIntersect.apply(this,args);};
for(let i=0;i<20;i++)clickCanvas.listeners.pointermove({...hoverPoint,pointerType:'mouse',buttons:0,isPrimary:true});
assert.equal(hoverRays,0,'pointer events only queue hover');assert.equal(raf.size,1,'pointer bursts coalesce into one frame');frame();
assert.equal(hoverRays,1,'one real raycast per frame');assert.equal(siteLabel.style.display,'','hover reveals existing module label');
assert.equal(clickCanvas.style.cursor,'pointer');assert.equal(siteLabel.style.color,'var(--accent)','hover is not warm selected state');
assert.equal(selected.length,0,'hover must not request exact selection/data');assert(camera.position.distanceTo(hoverCamera)<1e-8,'hover does not move camera');assert.equal(raf.size,0,'hover settles without an idle loop');
clickCanvas.listeners.pointerleave();frame();assert.equal(siteLabel.style.display,'none');assert.equal(clickCanvas.style.cursor,'');
clickCanvas.listeners.pointerdown({...hoverPoint,pointerId:7});clickCanvas.listeners.pointerleave();
// The release occurs on ownerDocument/outside the canvas; our pointerup is never called.
clickCanvas.listeners.pointermove({...hoverPoint,pointerId:7,pointerType:'mouse',buttons:0});frame();
assert.equal(siteLabel.style.display,'','hover recovers after dragging out and releasing outside the canvas');assert.equal(selected.length,0);
clickCanvas.listeners.pointerup({...hoverPoint,pointerId:7});assert.equal(selected.length,0,'leaving cancels the old click candidate');clickCanvas.listeners.pointerleave();frame();
clickCanvas.listeners.pointermove({...hoverPoint,pointerType:'mouse'});frame();clickCanvas.listeners.pointermove({...hoverPoint,pointerType:'touch'});frame();assert.equal(siteLabel.style.display,'none','touch clears any previous mouse hover');
clickCanvas.listeners.pointermove({...hoverPoint,pointerType:'mouse',isPrimary:false});frame();assert.equal(siteLabel.style.display,'none','secondary pointer must not preview');
clickCanvas.listeners.pointermove({clientX:-500,clientY:-500,pointerType:'mouse'});frame();assert.equal(siteLabel.style.display,'none','off-canvas pointer clears hover');
clickCanvas.listeners.pointerdown({...hoverPoint,pointerId:8});clickCanvas.listeners.pointermove({...hoverPoint,pointerId:8,pointerType:'mouse',buttons:0});frame();assert.equal(siteLabel.style.display,'','released-button movement recovers even when pointerup was missed');clickCanvas.listeners.pointerup({...hoverPoint,pointerId:8});assert.equal(selected.length,0,'recovered hover must not revive the missed gesture');clickCanvas.listeners.pointerleave();frame();
THREE.Raycaster.prototype.intersectObjects=originalIntersect;
const click=pointFor(sites[0]);clickCanvas.listeners.pointerdown({...click,button:2});clickCanvas.listeners.pointerup({...click,button:2});assert.equal(selected.length,0,'secondary button must not select');
clickCanvas.listeners.pointerdown({...click,pointerId:1});clickCanvas.listeners.pointerup({...click,pointerId:2});assert.equal(selected.length,0,'different pointer must not select');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointercancel();clickCanvas.listeners.pointerup(click);assert.equal(selected.length,0,'cancelled pointer must not select');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointerup(click);
assert.deepEqual(selected,['a'],'raycast returns exact captured site key');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointerup({...click,clientX:click.clientX+30});assert.equal(selected.length,1,'drag is not selection');
clickCanvas.listeners.pointerdown(click);clickCanvas.listeners.pointermove({...click,clientX:click.clientX+12,buttons:1});clickCanvas.listeners.pointermove({...click,buttons:1});clickCanvas.listeners.pointerup(click);assert.equal(selected.length,1,'a drag returning to its origin is not selection');frame();assert.equal(siteLabel.style.display,'none','drag does not preview');
let released=0;for(const o of objects){if(o.geometry)o.geometry.addEventListener('dispose',()=>released++);if(o.material)o.material.addEventListener('dispose',()=>released++);}
assert.equal(api.selectSite('a'),true);frame();frame();assert.equal(api.selectSite('a'),true);assert.equal(raf.size,0,'same-key UI synchronization must not restart focus');assert.equal(api.selectSite('missing'),false);
assert.equal(siteLabel.style.display,'','selected module label becomes readable');
assert.equal(sites[0].material.color.getHexString(),'eeddaa','selected actual site uses the warm selection token');
assert.equal(sites[1].material.color.getHexString(),'112233','unselected sites keep the neutral plane token');
clickCanvas.listeners.pointermove({...pointFor(sites[1]),pointerType:'pen',buttons:0});frame();
assert.equal(siteLabel.style.display,'','selection stays visible during another hover');assert.equal(siteLabel.style.color,'var(--selected)');
assert.equal(el.children[1].children[2].style.display,'','hovered second module becomes readable');assert.equal(sites[0].material.color.getHexString(),'eeddaa','hover cannot recolor selected site');assert.equal(selected.length,1,'hover cannot replace selected site');
clickCanvas.listeners.pointerleave();frame();assert.equal(el.children[1].children[2].style.display,'none');
api.selectSite('b');frame();assert.equal(siteLabel.style.display,'none','old module label hides when selection moves');
assert.equal(el.children[1].children[2].style.display,'','new selected module label is visible');
assert.equal(sites[0].material.color.getHexString(),'112233','previous selected site loses the warm fill');
api.selectSite(null);frame();assert.equal(el.children[1].children[2].style.display,'none','clear hides module labels');
api.selectSite('a');frame();
const selectedOpacity=sites[0].material.opacity,sceneObjects=scene.children.slice();camera.position.set(80,40,-20);camera.lookAt(new THREE.Vector3(0,0,0));api.resize();frame();camera.updateMatrixWorld();assert.deepEqual(scene.children,sceneObjects);assert.equal(sites[0].material.opacity,selectedOpacity,'reset framing preserves selected-site highlight');for(const plane of planes)for(const x of [-.45,.45])for(const y of [-.825,.825]){const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(camera);assert((p.x+1)*800>=360&&(p.x+1)*800<=1040);assert((1-p.y)*450>=0&&(1-p.y)*450<=900);}assert.equal(selected.length,1,'reset does not select or read a site');
assert.throws(()=>api.setLayers([{key:'x',sites:[{key:'same'},{key:'same'}]}]),/unique/i);
assert.throws(()=>api.setLayers(Array.from({length:13},(_,i)=>({key:String(i),sites:[]}))),/12/);
clickCanvas.listeners.pointerdown(click);
api.setLayers([{key:'new',label:'Replacement',sites:[{key:'fresh',label:'Fresh'}]}]);assert.equal(api.selectSite('a'),false);clickCanvas.listeners.pointerup(click);assert.equal(selected.length,1,'replacement invalidates in-flight pointer');assert(released>0,'replaced geometry and material disposed');frame();
assert.equal(raf.size,0,'reduced motion settles without an idle render loop');
let freshSite;scene.traverse(o=>{if(o.userData.siteKey==='fresh')freshSite=o;});
const freshHover=pointFor(freshSite);clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});
api.setLayers([{key:'new',label:'Replacement',sites:[{key:'fresh',label:'Fresh'}]}]);frame();assert.equal(el.children[1].children[1].style.display,'none','queued hover cannot cross layer replacement even when keys repeat');assert.equal(clickCanvas.style.cursor,'');
function cancelledGestureBoundary(hide,show,label){
 const count=selected.length;clickCanvas.listeners.pointerdown({...freshHover,pointerId:9});hide();show();frame();
 clickCanvas.listeners.pointerup({...freshHover,pointerId:9});assert.equal(selected.length,count,label+' cancels the old selection gesture');
 clickCanvas.listeners.pointermove({...freshHover,pointerId:9,pointerType:'mouse',buttons:0});frame();assert.equal(clickCanvas.style.cursor,'pointer',label+' still permits fresh hover');clickCanvas.listeners.pointerleave();frame();
}
cancelledGestureBoundary(()=>api.setActive(false),()=>api.setActive(true),'inactive scene');
cancelledGestureBoundary(()=>{document.hidden=true;document.listeners.visibilitychange();},()=>{document.hidden=false;document.listeners.visibilitychange();},'hidden document');
cancelledGestureBoundary(()=>api.resize(),()=>{},'resize');
clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});frame();assert.equal(clickCanvas.style.cursor,'pointer');
api.resize();frame();assert.equal(clickCanvas.style.cursor,'','resizing invalidates pointer projection');
clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});frame();
document.hidden=true;document.listeners.visibilitychange();assert.equal(clickCanvas.style.cursor,'','hidden document clears hover');assert.equal(raf.size,0);clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});assert.equal(raf.size,0,'background documents cannot schedule hover');document.hidden=false;document.listeners.visibilitychange();frame();assert.equal(el.children[1].children[1].style.display,'none','hover does not revive on visibility');
clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});frame();api.setActive(false);assert.equal(raf.size,0);assert.equal(clickCanvas.style.cursor,'');clickCanvas.listeners.pointermove({...freshHover,pointerType:'mouse'});assert.equal(raf.size,0,'hidden scene does no hover work');api.setActive(true);frame();assert.equal(el.children[1].children[1].style.display,'none');api.resize();
const canvas=el.children.find(e=>e.listeners.webglcontextlost);assert(canvas);canvas.listeners.pointermove({...freshHover,pointerType:'mouse'});frame();canvas.listeners.webglcontextlost({preventDefault(){}});assert.equal(canvas.style.cursor,'','lost WebGL clears hover cursor');assert.equal(fail.length,1);assert.equal(raf.size,0);assert.equal(api.selectSite('fresh'),false);
api.dispose();api.dispose();assert.equal(disposed,1);assert.equal(el.children.length,0);
// Framing and actual raycast selection across layer counts and viewport geometries.
for(const count of [1,2,12]) for(const [width,height,left,right] of [[1600,900,360,560],[800,900,0,0],[390,844,0,0]]){
 const box=new El();box.clientWidth=width;box.clientHeight=height;const picks=[];
 const instance=context.MLGScene.create(box,{getViewInsets:()=>({left,right}),onSelect:k=>picks.push(k)});
 instance.setLayers(Array.from({length:count},(_,i)=>({key:'layer'+i,sites:[{key:'site'+i}]})));frame();
 const view=renders.at(-1),objs=[];view.scene.traverse(o=>objs.push(o));view.scene.updateMatrixWorld(true);view.camera.updateMatrixWorld();
 const layerPlanes=objs.filter(o=>o.userData.kind==='layer'),targets=objs.filter(o=>o.userData.siteKey);
 layerPlanes.forEach((plane,i)=>{assert(Math.abs(plane.position.x+i*.30)<1e-9);assert(Math.abs(plane.position.y-i*.08)<1e-9);assert(Math.abs(plane.position.z+i*1.20)<1e-9);});
 for(const plane of layerPlanes) for(const x of [-.45,.45]) for(const y of [-.825,.825]){
  const p=new THREE.Vector3(x,y,0).applyMatrix4(plane.matrixWorld).project(view.camera),sx=(p.x+1)*width/2,sy=(1-p.y)*height/2;
  assert(sx>=left-1e-6 && sx<=width-right+1e-6 && sy>=-1e-6 && sy<=height+1e-6,`layer outside viewport: ${count}, ${width}, ${sx}, ${sy}`);
 }
 if(count>1){
  const size=plane=>{const a=new THREE.Vector3(0,-.825,0).applyMatrix4(plane.matrixWorld).project(view.camera),b=new THREE.Vector3(0,.825,0).applyMatrix4(plane.matrixWorld).project(view.camera);return Math.abs(a.y-b.y);};
  assert(size(layerPlanes[0])>size(layerPlanes.at(-1)),'ordered layers visibly recede');
 }
 const canvas=box.children.find(e=>e.listeners.webglcontextlost);canvas.clientWidth=width;canvas.clientHeight=height;
 for(const target of targets){const p=target.getWorldPosition(new THREE.Vector3()).project(view.camera),point={clientX:(p.x+1)*width/2,clientY:(1-p.y)*height/2};canvas.listeners.pointerdown(point);canvas.listeners.pointerup(point);}
 assert.deepEqual(picks,targets.map(t=>t.userData.siteKey),`each layer remains directly pickable: ${count}, ${width}`);
 instance.dispose();
}
// Earlier planes, site hit targets and labels retire when navigating deeper.
for(const count of [1,2,12]){
 const holder=new El(),reports=[],picks=[];const view=context.MLGScene.create(holder,{onNavigate:n=>reports.push(n),onSelect:k=>picks.push(k)});
 view.setLayers(Array.from({length:count},(_,i)=>({key:'depth'+i,label:'Depth '+i,sites:[{key:'depth-site'+i}]})));frame();
 const screen=renders.at(-1),targets=[];screen.scene.traverse(o=>{if(o.userData.siteKey)targets.push(o);});
 if(count>1){assert(view.selectSite('depth-site0'));frame();assert(view.goToLayer(count-1));frame();assert.equal(targets[0].parent.visible,false);assert.equal(targets.at(-1).parent.visible,true);assert.equal(holder.children[1].children[0].style.display,'none','earlier layer label hidden');assert.equal(holder.children[1].children[1].style.display,'none','earlier selected site label hidden');
  const oldIntersect=THREE.Raycaster.prototype.intersectObjects;let offered=[];THREE.Raycaster.prototype.intersectObjects=function(items,...rest){offered=items;return oldIntersect.call(this,items,...rest);};
  const canvas=holder.children.find(e=>e.listeners.wheel),p=targets.at(-1).getWorldPosition(new THREE.Vector3()).project(screen.camera),point={clientX:(p.x+1)*800,clientY:(1-p.y)*450};canvas.listeners.pointerdown(point);canvas.listeners.pointerup(point);THREE.Raycaster.prototype.intersectObjects=oldIntersect;
  assert(!offered.includes(targets[0]) && offered.includes(targets.at(-1)),'hidden targets are excluded from raycasting');assert(!picks.includes('depth-site0'));
  assert(view.selectSite('depth-site0'));frame();assert.equal(reports.at(-1).index,0,'list selection returns to the earlier depth');assert(targets[0].parent.visible);assert.equal(targets[0].material.color.getHexString(),'eeddaa','selection preserved');
  assert(view.goToLayer(count-1));frame();
 }
 view.resize();frame();screen.camera.updateMatrixWorld();screen.scene.updateMatrixWorld(true);const current=targets.at(-1),proj=current.getWorldPosition(new THREE.Vector3()).project(screen.camera);assert(current.parent.visible && Math.abs(proj.x)<=1 && Math.abs(proj.y)<=1,'current site visible and bounded after resize');
 if(count>1){assert(view.goToLayer(0));frame();assert(targets.every(t=>t.parent.visible),'Home depth restores all layers');}
 view.dispose();
}
// Navigation input keeps selections stable, clamps boundaries and respects user intent.
const navEl=new El(),navReports=[],nav=context.MLGScene.create(navEl,{onNavigate:n=>navReports.push(n)});
nav.setLayers(Array.from({length:4},(_,i)=>({key:'n'+i,label:'Layer '+i,sites:[{key:'n-site'+i}]})));frame();
const navCanvas=navEl.children.find(e=>e.listeners.wheel),navControls=latestControls;
assert.equal(navCanvas.attributes.tabindex,'0');assert.equal(navCanvas.attributes.role,'region');assert(navCanvas.attributes['aria-label'].includes('arrows'));
assert(nav.selectSite('n-site0'));frame();const navScene=renders.at(-1).scene,navTargets=[];navScene.traverse(o=>{if(o.userData.siteKey)navTargets.push(o);});
assert.equal(nav.navigateLayers(-1),false);assert.equal(nav.goToLayer(99),true);assert.equal(navReports.at(-1).index,3);assert.equal(nav.navigateLayers(1),false);assert.equal(nav.goToLayer(-10),true);assert.equal(navReports.at(-1).index,0);frame();assert.equal(navTargets[0].material.color.getHexString(),'eeddaa','navigation retains selection');
assert(nav.setMode('orbit'));assert(navControls.enableRotate);assert(!navControls.enablePan);assert(nav.setMode('pan'));assert(navControls.enablePan);assert(!navControls.enableRotate);assert(nav.setMode('select'));assert(!navControls.enablePan && !navControls.enableRotate);assert.equal(nav.setMode('invalid'),false);frame();
let prevented=0;const wheel=(dy,extra={})=>navCanvas.listeners.wheel({deltaY:dy,deltaX:0,deltaMode:0,preventDefault(){prevented++;},...extra});
wheel(20,{ctrlKey:true});assert.equal(prevented,0,'modified zoom is untouched');wheel(20,{deltaX:30});assert.equal(prevented,0,'horizontal scroll is untouched');wheel(30);wheel(30);assert.equal(navReports.at(-1).index,0,'small deltas accumulate');assert.equal(prevented,0,'subthreshold scroll passes through');wheel(25);assert.equal(navReports.at(-1).index,1,'accumulated scroll advances once');assert.equal(prevented,1,'only navigation consumes scroll');frame();wheel(100);assert.equal(navReports.at(-1).index,1,'burst is rate limited');assert.equal(prevented,1,'throttled scroll passes through');now+=250;wheel(1);assert.equal(navReports.at(-1).index,2,'accumulated burst advances after interval');frame();now+=250;wheel(5,{deltaMode:1});assert.equal(navReports.at(-1).index,3,'line deltas normalize');frame();const beforeBoundary=prevented;wheel(100);assert.equal(prevented,beforeBoundary,'boundary does not trap page scroll');
const key=(key,extra={})=>navCanvas.listeners.keydown({key,preventDefault(){prevented++;},...extra});key('Home',{shiftKey:true});assert.equal(navReports.at(-1).index,3);key('Home');assert.equal(navReports.at(-1).index,0);key('End');assert.equal(navReports.at(-1).index,3);key('ArrowUp');assert.equal(navReports.at(-1).index,2);frame();frame();assert.equal(raf.size,0,'reduced motion settles without idle loop');assert(nav.fitView());assert.equal(navReports.at(-1).index,0);frame();nav.dispose();
// Animated focus uses real OrbitControls and a 600 ms transition when motion is permitted.
reduced=false;now=0;
const movingEl=new El(),moving=context.MLGScene.create(movingEl,{});
const fullPage=Array.from({length:12},(_,i)=>({key:'layer'+i,sites:Array.from({length:6},(_,j)=>({key:`s${i}-${j}`}))}));
assert.doesNotThrow(()=>moving.setLayers(fullPage));frame();let bounded=[];renders.at(-1).scene.traverse(o=>bounded.push(o));assert.equal(bounded.filter(o=>o.userData.siteKey).length,72);assert.equal(bounded.filter(o=>o.userData.kind==='layer').length,12);assert.equal(bounded.filter(o=>o.geometry).length,240,'fixed resource bound: two geometries per layer and three per site');assert.equal(bounded.filter(o=>o.geometry && o.geometry.type==='CircleGeometry').length,72,'one marker for each captured site');
fullPage[0].sites.push({key:'site73'});assert.throws(()=>moving.setLayers(fullPage),/72/);
moving.setLayers([{key:'animated',label:'L'.repeat(200),sites:[{key:'focus',label:'M'.repeat(200)},{key:'other'}]},{key:'second',sites:[{key:'second-site'}]}]);assert(movingEl.children[1].children.every(e=>e.textContent.length<=64),'visual labels must remain bounded');frame();
const movingCamera=renders.at(-1).camera,before=movingCamera.position.clone();
moving.selectSite('focus');assert(moving.navigateLayers(1));frame(300);assert(raf.size>0,'navigation schedules continuation');frame(600);frame(601);assert(movingCamera.position.distanceTo(before)>0,'navigation moves camera');assert.equal(raf.size,0,'completed motion has no idle loop');
assert(moving.navigateLayers(-1));frame(200);moving.resize();const resized=movingCamera.position.clone();frame(600);assert(movingCamera.position.distanceTo(resized)<1e-8,'resize cancels animation and frames current layer');assert.equal(moving.selectSite('focus'),true);
// Hidden documents pause work; visibility resumes a single requested update.
document.hidden=true;document.listeners.visibilitychange();assert.equal(raf.size,0);document.hidden=false;document.listeners.visibilitychange();assert(raf.size>0);frame(602);
moving.dispose();assert.equal(movingEl.children.length,0);
constructFailure=true;const missingEl=new El(),missingReasons=[];
const missing=context.MLGScene.create(missingEl,{onUnavailable:r=>missingReasons.push(r)});assert.equal(missingReasons.length,1);assert.equal(missing.selectSite('x'),false);missing.dispose();assert.equal(missingEl.children.length,0);constructFailure=false;
const brokenEl=new El(),brokenReasons=[],broken=context.MLGScene.create(brokenEl,{onUnavailable:r=>brokenReasons.push(r)});renderFailure=true;frame();assert.equal(brokenReasons.length,1);assert.equal(raf.size,0);broken.dispose();renderFailure=false;
console.log('app_scene_contract: PASS (real Three math; fake renderer, not GPU proof)');


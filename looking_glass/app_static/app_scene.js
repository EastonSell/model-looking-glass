/* Bounded captured-site scene. No tensors, inferred neurons, or causal edges. */
(function (global) {
  'use strict';
  function create(container, options) {
    options = options || {};
    const T = global.THREE;
    const style = global.getComputedStyle ? global.getComputedStyle(container) : null;
    const token = (name, fallback) => style && style.getPropertyValue(name).trim() || fallback;
    const palette = {ice:token('--accent','#bfe6ff'),neutral:token('--muted','#9aa0b0'),selected:token('--selected','#eed9a0')};
    let renderer, controls, dead = false, failed = false, active = true, raf = 0;
    let content, keys = new Map(), labels = [], layerGroups = [], down = null, generation = 0, selected = null, hovered = null, hoverPoint = null;
    let width = 1, height = 1, animation = null, layerIndex = 0, layerLabels = [], mode = 'select', wheelAccum = 0, wheelLastStep = -Infinity;
    const layerPosition = i => new T.Vector3(-i*0.30,i*0.08,-i*1.20);
    const media = global.matchMedia ? global.matchMedia('(prefers-reduced-motion: reduce)') : {matches:true};
    const scene = T ? new T.Scene() : null;
    const camera = T ? new T.PerspectiveCamera(35, 1, 0.01, 200) : null;
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {position:'absolute',inset:'0',pointerEvents:'none',overflow:'hidden'});
    overlay.setAttribute('aria-hidden','true');
    function stop() { if (raf) global.cancelAnimationFrame(raf); raf = 0; }
    function unavailable(reason) {
      if (dead || failed) return;
      failed = true; stop(); releaseContent();
      if (controls) controls.enabled = false;
      if (renderer) renderer.domElement.style.display = 'none';
      overlay.style.display = 'none';
      if (options.onUnavailable) options.onUnavailable(reason);
    }
    function schedule() { if (!dead && !failed && active && !document.hidden && !raf) raf = global.requestAnimationFrame(draw); }
    function finishAnimation() {
      if(!animation)return;
      camera.position.copy(animation.camera).add(animation.to.clone().sub(animation.from));
      controls.target.copy(animation.to);animation=null;
    }
    function styleSites() {
      for(const [key,item] of keys){
        const chosen=key===selected,preview=key===hovered;
        item.outline.material.color.set(chosen?palette.selected:preview?palette.ice:palette.neutral);
        item.target.material.color.set(chosen?palette.selected:palette.ice);
        item.target.material.opacity=0;
        item.marker.material.color.set(chosen?palette.selected:preview?palette.ice:palette.neutral);
        item.marker.material.opacity=chosen?1:preview?0.95:0.72;
        item.outline.material.opacity=0;
      }
    }
    function setHovered(key) {
      if(hovered===key)return false;
      hovered=key;styleSites();if(renderer)renderer.domElement.style.cursor=key?'pointer':'';return true;
    }
    function clearHover() { hoverPoint=null;if(setHovered(null))schedule(); }
    function hitSite(point) {
      const rect=renderer.domElement.getBoundingClientRect();
      if(!rect.width || !rect.height || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x<rect.left || point.x>rect.left+rect.width || point.y<rect.top || point.y>rect.top+rect.height)return null;
      const ray=new T.Raycaster();camera.updateMatrixWorld();scene.updateMatrixWorld(true);
      ray.setFromCamera(new T.Vector2((point.x-rect.left)/rect.width*2-1,-(point.y-rect.top)/rect.height*2+1),camera);
      const hit=ray.intersectObjects(Array.from(keys.values()).filter(v=>v.layerIndex>=layerIndex).map(v=>v.target),false)[0];
      return hit && keys.has(hit.object.userData.siteKey)?hit.object.userData.siteKey:null;
    }
    function draw(now) {
      raf = 0;
      if (dead || failed || !active || document.hidden) return;
      if (animation) {
        const t = Math.min(1, Math.max(0, (now-animation.start)/600)), eased = 1-Math.pow(1-t,3);
        controls.target.lerpVectors(animation.from, animation.to, eased);
        camera.position.copy(animation.camera).add(controls.target.clone().sub(animation.from));
        if (t === 1) animation = null;
      }
      controls.update(); camera.updateMatrixWorld();
      const preview=hoverPoint;hoverPoint=null;
      if(preview)setHovered(!down && !animation && preview.generation===generation?hitSite(preview):null);
      for (const label of labels) {
        const p = label.position.clone().project(camera);
        label.element.style.display = label.layerIndex < layerIndex || (label.siteKey && label.siteKey !== selected && label.siteKey !== hovered) || p.z < -1 || p.z > 1 ? 'none' : '';
        if(label.siteKey)label.element.style.color=label.siteKey===selected?'var(--selected)':'var(--accent)';
        label.element.style.left = ((p.x+1)*width/2)+'px';
        label.element.style.top = ((1-p.y)*height/2)+'px';
      }
      try { renderer.render(scene,camera); } catch (error) { unavailable('WebGL rendering failed. Use the captured-site list.'); return; }
      if (animation) schedule();
    }
    function releaseContent() {
      if (content) { content.traverse(o => { if(o.geometry)o.geometry.dispose(); if(o.material) { const ms=Array.isArray(o.material)?o.material:[o.material]; ms.forEach(m=>m.dispose()); } }); scene.remove(content); }
      for (const label of labels) label.element.remove();
      labels = []; layerGroups = []; keys.clear(); content = null; animation = null; down = null; hovered=null;hoverPoint=null;if(renderer)renderer.domElement.style.cursor='';
    }
    function label(text, position, corner, siteKey=null, layer=0) {
      const element=document.createElement('span'); element.textContent=text.length>64?text.slice(0,63)+'…':text; element.title=text;
      Object.assign(element.style,{position:'absolute',font:'400 11px "JetBrains Mono", monospace',color:siteKey?'var(--selected)':'var(--text-3)',whiteSpace:'nowrap',transform:corner?'translate(6px, 6px)':'translate(8px, -50%)'});
      overlay.appendChild(element); labels.push({element,position,siteKey,layerIndex:layer});
    }
    function validate(layers) {
      if (!Array.isArray(layers) || layers.length > 12) throw new Error('Scene accepts at most 12 captured layers.');
      const layerKeys=new Set(),siteKeys=new Set();
      for(const layer of layers) {
        if(!layer || typeof layer.key!=='string' || !layer.key || layerKeys.has(layer.key)) throw new Error('Layer keys must be unique nonempty strings.');
        layerKeys.add(layer.key);
        if(!Array.isArray(layer.sites)) throw new Error('Each layer requires captured sites.');
        for(const site of layer.sites) {
          if(!site || typeof site.key!=='string' || !site.key || siteKeys.has(site.key)) throw new Error('Site keys must be unique nonempty strings.');
          siteKeys.add(site.key);
        }
      }
      if(siteKeys.size>72) throw new Error('Scene accepts at most 72 captured sites.');
    }
    function setLayers(layers) {
      validate(layers);
      if(dead || failed)return;
      generation++; selected=null; releaseContent(); layerIndex=0; layerLabels=layers.map(layer=>String(layer.label || layer.key)); content=new T.Group(); scene.add(content);
      layers.forEach((layer,i)=>{
        const group=new T.Group();content.add(group);layerGroups.push(group);
        // Each plane is one captured layer; ordered placement supplies perspective only.
        const x=-i*0.30,yLayer=i*0.08,z=-i*1.20;
        const plane=new T.Mesh(new T.PlaneGeometry(0.90,1.65),new T.MeshBasicMaterial({color:palette.ice,opacity:0.025,transparent:true,depthWrite:false,side:T.DoubleSide}));
        plane.position.set(x,yLayer,z); plane.userData.kind='layer'; group.add(plane);
        const edge=new T.LineSegments(new T.EdgesGeometry(plane.geometry),new T.LineBasicMaterial({color:palette.neutral,transparent:true,opacity:0.35,depthWrite:false}));edge.position.copy(plane.position);group.add(edge);
        label(String(layer.label || layer.key),new T.Vector3(x-0.45,yLayer+0.825,z),true,null,i);
        // One marker per captured site. Grid placement is navigation layout only;
        // it does not encode tensor dimensions, neurons, or connections.
        const count=layer.sites.length, columns=Math.max(1,Math.ceil(Math.sqrt(count))), rows=Math.max(1,Math.ceil(count/columns));
        const cellWidth=0.76/columns, cellHeight=1.42/rows;
        layer.sites.forEach((site,j)=>{
          const column=j%columns, row=Math.floor(j/columns);
          const sx=x-0.38+(column+0.5)*cellWidth, y=yLayer+0.71-(row+0.5)*cellHeight;
          const target=new T.Mesh(new T.PlaneGeometry(cellWidth*0.92,cellHeight*0.92),new T.MeshBasicMaterial({color:palette.ice,transparent:true,opacity:0,depthWrite:false,side:T.DoubleSide}));
          target.position.set(sx,y,z+0.004);target.userData.siteKey=site.key;group.add(target);
          const outline=new T.LineSegments(new T.EdgesGeometry(target.geometry),new T.LineBasicMaterial({color:palette.neutral,transparent:true,opacity:0,depthWrite:false}));outline.position.copy(target.position);group.add(outline);
          const marker=new T.Mesh(new T.CircleGeometry(0.016,12),new T.MeshBasicMaterial({color:palette.neutral,transparent:true,opacity:0.72,depthWrite:false,side:T.DoubleSide}));
          marker.position.set(sx,y,z+0.008);group.add(marker);
          keys.set(site.key,{target,outline,marker,layerIndex:i});label(String(site.label || site.key),target.position.clone().add(new T.Vector3(0,0,0.012)),false,site.key,i);
        });
      });
      resize(); reportNavigation();
    }
    function resize() {
      if(dead || failed)return;
      pointerCancel();
      const rect=container.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);
      renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1,2));renderer.setSize(width,height,false);
      const insets=options.getViewInsets?options.getViewInsets():null;
      let left=Number.isFinite(insets?.left)?Math.max(0,insets.left):0,right=Number.isFinite(insets?.right)?Math.max(0,insets.right):0;
      if(left+right>=width){left=0;right=0;} // Ignore stale/non-overlapping layout measurements.
      const gap=width-left-right,center=left+gap/2;
      camera.aspect=width/height;camera.setViewOffset(width,height,width/2-center,0,width,height);
      // Frame real plane corners from the current layer forward, avoiding the
      // nonexistent corners of a world-axis bounding box.
      const target=layerPosition(layerIndex);
      const direction=new T.Vector3(Math.sin(42*Math.PI/180)*Math.cos(7*Math.PI/180),Math.sin(7*Math.PI/180),Math.cos(42*Math.PI/180)*Math.cos(7*Math.PI/180));
      camera.position.copy(target).add(direction);camera.lookAt(target);camera.updateMatrixWorld();
      const inverse=camera.quaternion.clone().invert();let distance=2;
      const tan=Math.tan(35*Math.PI/360);
      for(let i=layerIndex;i<Math.max(layerLabels.length,layerIndex+1);i++){
        const center=layerPosition(i);
        for(const x of [-0.45,0.45])for(const y of [-0.825,0.825]){
          const p=center.clone().add(new T.Vector3(x,y,0)).sub(target).applyQuaternion(inverse);
          distance=Math.max(distance,p.z+Math.abs(p.x)/(tan*(gap/height)*0.82),p.z+Math.abs(p.y)/(tan*0.80));
        }
      }
      camera.position.copy(target).addScaledVector(direction,distance);controls.target.copy(target);
      controls.minDistance=distance*0.45;controls.maxDistance=distance*2.5;camera.far=Math.max(200,distance*5);camera.updateProjectionMatrix();animation=null;controls.update();schedule();
    }
    function selectSite(key) {
      if(dead || failed)return false;
      const entry=keys.get(key);
      if(key!==null && !entry)return false;
      if(entry && entry.layerIndex<layerIndex)goToLayer(entry.layerIndex);
      if(selected===key)return true;
      clearHover();finishAnimation();selected=key;styleSites();
      schedule();return true;
    }
    function pointerDown(event){if(mode!=='select')return;clearHover();down=null;if(event.isPrimary===false || (event.button!==undefined && event.button!==0))return;down={x:event.clientX,y:event.clientY,id:event.pointerId,generation,moved:false};}
    function pointerMove(event){
      if(mode!=='select')return;
      if(down && down.id===event.pointerId && event.buttons===0)down=null;
      if(down){if(down.id===event.pointerId && Math.hypot(event.clientX-down.x,event.clientY-down.y)>5)down.moved=true;clearHover();return;}
      if(dead || failed || !active || document.hidden || animation || event.pointerType==='touch' || event.isPrimary===false || event.buttons){clearHover();return;}
      hoverPoint={x:event.clientX,y:event.clientY,generation};schedule();
    }
    function pointerCancel(){down=null;clearHover();}
    function pointerUp(event){
      if(mode!=='select')return;
      const start=down;down=null;
      if(!start || start.id!==event.pointerId || event.isPrimary===false || (event.button!==undefined && event.button!==0) || start.generation!==generation || start.moved || !active || failed || dead || Math.hypot(event.clientX-start.x,event.clientY-start.y)>5)return;
      const key=hitSite({x:event.clientX,y:event.clientY});
      if(key && options.onSelect)options.onSelect(key);
    }
    function syncLayerVisibility(){layerGroups.forEach((group,i)=>{group.visible=i>=layerIndex;});}
    function reportNavigation(){if(options.onNavigate)options.onNavigate({index:layerIndex,total:layerLabels.length,label:layerLabels[layerIndex] || ''});}
    function goToLayer(index){
      if(dead || failed || !active || !layerLabels.length || !Number.isFinite(index))return false;
      const next=Math.max(0,Math.min(layerLabels.length-1,Math.trunc(index)));
      if(next===layerIndex)return false;
      const delta=layerPosition(next).sub(layerPosition(layerIndex));layerIndex=next;syncLayerVisibility();pointerCancel();
      // Advance from the current rendered position, even during an earlier animation.
      const from=controls.target.clone(), to=(animation ? animation.to.clone() : from.clone()).add(delta);
      if(media.matches){controls.target.copy(to);camera.position.add(to.clone().sub(from));animation=null;}
      else animation={start:global.performance.now(),from,to,camera:camera.position.clone()};
      reportNavigation();schedule();return true;
    }
    function navigateLayers(delta){return Number.isFinite(delta) && delta!==0 ? goToLayer(layerIndex+(delta>0?1:-1)) : false;}
    function setMode(value){
      if(!['select','orbit','pan'].includes(value) || dead || failed)return false;
      mode=value;pointerCancel();finishAnimation();renderer.domElement.style.touchAction=value==='select'?'pan-y':'none';
      controls.enableRotate=value==='orbit';controls.enablePan=value==='pan';
      if(T.MOUSE)controls.mouseButtons.LEFT=value==='pan'?T.MOUSE.PAN:T.MOUSE.ROTATE;
      schedule();return true;
    }
    function wheel(event){
      if(dead || failed || !active || document.hidden || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || Math.abs(event.deltaX)>=Math.abs(event.deltaY) || !event.deltaY)return;
      const direction=event.deltaY>0?1:-1;
      if(!layerLabels.length || (direction<0 && layerIndex===0) || (direction>0 && layerIndex===layerLabels.length-1)){wheelAccum=0;return;}
      const pixels=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?height:1);
      if(Math.sign(wheelAccum)!==direction)wheelAccum=0;
      wheelAccum=Math.max(-160,Math.min(160,wheelAccum+pixels));
      const now=global.performance.now();
      if(Math.abs(wheelAccum)>=80 && now-wheelLastStep>=200){
        if(navigateLayers(direction)){event.preventDefault();wheelLastStep=now;wheelAccum=0;}
      }
    }
    function keydown(event){
      if(dead || failed || !active || document.hidden || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey)return;
      let changed=false;
      if(event.key==='ArrowDown' || event.key==='PageDown')changed=navigateLayers(1);
      else if(event.key==='ArrowUp' || event.key==='PageUp')changed=navigateLayers(-1);
      else if(event.key==='Home')changed=goToLayer(0);
      else if(event.key==='End')changed=goToLayer(layerLabels.length-1);
      if(changed)event.preventDefault();
    }
    function fitView(){if(dead || failed)return false;layerIndex=0;wheelAccum=0;syncLayerVisibility();resize();reportNavigation();return true;}
    function lost(event){event.preventDefault();unavailable('WebGL context lost. Use the captured-site list.');}
    function visibility(){stop();finishAnimation();pointerCancel();schedule();}
    function motion(){finishAnimation();clearHover();controls.enableDamping=!media.matches;schedule();}
    function interaction(){finishAnimation();clearHover();schedule();}
    function cameraChanged(){clearHover();schedule();}
    function setActive(value){active=!!value;if(!active)pointerCancel();if(controls)controls.enabled=active&&!failed;if(renderer)renderer.domElement.style.display=active&&!failed?'':'none';overlay.style.display=active&&!failed?'':'none';stop();if(active)schedule();}
    function dispose(){if(dead)return;dead=true;stop();releaseContent();document.removeEventListener('visibilitychange',visibility);if(media.removeEventListener)media.removeEventListener('change',motion);if(controls)controls.dispose();if(renderer){renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointerup',pointerUp);renderer.domElement.removeEventListener('pointercancel',pointerCancel);renderer.domElement.removeEventListener('pointermove',pointerMove);renderer.domElement.removeEventListener('pointerleave',pointerCancel);renderer.domElement.removeEventListener('webglcontextlost',lost);renderer.domElement.removeEventListener('wheel',wheel);renderer.domElement.removeEventListener('keydown',keydown);renderer.dispose();renderer.domElement.remove();}overlay.remove();}
    const api={setLayers,selectSite,setActive,resize,dispose,setMode,navigateLayers,goToLayer,fitView};
    try {
      if(!T || !T.OrbitControls)throw new Error('Local Three.js assets unavailable.');
      renderer=new T.WebGLRenderer({alpha:true,antialias:true});
      Object.assign(renderer.domElement.style,{position:'absolute',inset:'0',width:'100%',height:'100%',touchAction:'none'});
      renderer.domElement.setAttribute('tabindex','0');renderer.domElement.setAttribute('role','region');renderer.domElement.setAttribute('aria-label','Captured layers. Use up and down arrows to move through layers.');
      container.appendChild(renderer.domElement);container.appendChild(overlay);
      controls=new T.OrbitControls(camera,renderer.domElement);renderer.domElement.style.touchAction='pan-y';controls.enablePan=false;controls.enableRotate=false;controls.enableZoom=false;controls.enableDamping=!media.matches;controls.dampingFactor=0.08;
      controls.addEventListener('change',cameraChanged);controls.addEventListener('start',interaction);
      renderer.domElement.addEventListener('wheel',wheel,{passive:false});renderer.domElement.addEventListener('keydown',keydown);renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp);renderer.domElement.addEventListener('pointercancel',pointerCancel);renderer.domElement.addEventListener('pointermove',pointerMove);renderer.domElement.addEventListener('pointerleave',pointerCancel);renderer.domElement.addEventListener('webglcontextlost',lost);
      document.addEventListener('visibilitychange',visibility);if(media.addEventListener)media.addEventListener('change',motion);
      resize();
    } catch(error){unavailable('WebGL unavailable. Use the captured-site list.');}
    return api;
  }
  global.MLGScene={create};
})(window);

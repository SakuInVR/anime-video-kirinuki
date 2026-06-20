import { SamModel, SamProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm';
env.allowLocalModels = false;

const $ = s => document.querySelector(s);
const fileInput=$('#fileInput'), video=$('#video'), canvas=$('#canvas');
const ctx=canvas.getContext('2d',{willReadFrequently:true});
const controls={duration:$('#duration'),softness:$('#softness'),fps:$('#fps')};
let file,selection=null,dragStart=null,frameImage=null,busy=false,model=null,processor=null;

$('#pickButton').onclick=()=>fileInput.click();
$('#changeButton').onclick=()=>fileInput.click();
fileInput.onchange=()=>loadFile(fileInput.files[0]);
const dz=$('#dropZone');
['dragenter','dragover'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.add('drag')}));
['dragleave','drop'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.remove('drag')}));
dz.ondrop=e=>loadFile(e.dataTransfer.files[0]);

const once=(el,event)=>new Promise(r=>el.addEventListener(event,r,{once:true}));
async function loadFile(f){
  if(!f||!f.type.startsWith('video/'))return alert('動画ファイルを選んでください。');
  file=f;video.src=URL.createObjectURL(f);await once(video,'loadedmetadata');
  canvas.width=Math.min(video.videoWidth,960);canvas.height=Math.round(canvas.width*video.videoHeight/video.videoWidth);
  $('#fileName').textContent=f.name;$('#videoMeta').textContent=`${video.videoWidth}×${video.videoHeight} / ${video.duration.toFixed(1)}秒`;
  $('#dropZone').classList.add('hidden');$('#workspace').classList.remove('hidden');selection=null;await seek(0);
}
async function seek(t){video.currentTime=Math.max(0,Math.min(t,video.duration-.01));await once(video,'seeked');drawFrame()}
function drawFrame(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(video,0,0,canvas.width,canvas.height);frameImage=ctx.getImageData(0,0,canvas.width,canvas.height);if(selection){ctx.strokeStyle='#ff5c8a';ctx.lineWidth=3;ctx.setLineDash([8,5]);ctx.strokeRect(selection.x,selection.y,selection.w,selection.h);ctx.setLineDash([])}}
$('#timeline').oninput=()=>seek(video.duration*$('#timeline').value/1000);
function pos(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height}}
canvas.onpointerdown=e=>{dragStart=pos(e);selection=null;canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!dragStart)return;const p=pos(e);selection={x:Math.min(p.x,dragStart.x),y:Math.min(p.y,dragStart.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};drawFrame()};
canvas.onpointerup=()=>{dragStart=null;if(selection?.w>12&&selection?.h>12){$('#hint').textContent='選択できました。輪郭AIでプレビューしてください';$('#previewButton').disabled=$('#exportButton').disabled=false}};

async function loadSam(){
  if(model)return;
  const status=$('#modelStatus');status.textContent='輪郭AIを読み込み中…（初回のみ）';
  try{
    model=await SamModel.from_pretrained('Xenova/slimsam-77-uniform',{dtype:'q8',progress_callback:p=>{if(p.status==='progress')status.textContent=`輪郭AIを読み込み中… ${Math.round(p.progress||0)}%`}});
    processor=await SamProcessor.from_pretrained('Xenova/slimsam-77-uniform');
    status.textContent='輪郭AIの準備完了';status.className='model-note ready';
  }catch(e){model=null;status.textContent='モデルを読み込めませんでした。通信状態を確認してください';status.className='model-note error';throw e}
}
async function inferMask(image,sel){
  await loadSam();
  const temp=document.createElement('canvas');temp.width=image.width;temp.height=image.height;temp.getContext('2d').putImageData(image,0,0);
  const raw=RawImage.fromCanvas(temp);
  // A single point often makes SAM interpret an anime character's shirt or
  // hair as the whole object. Spread positive prompts across the selected
  // figure so the requested object is the complete character.
  const points=[.16,.34,.52,.70,.86].map(y=>[sel.x+sel.w*.5,sel.y+sel.h*y]);
  points.push([sel.x+sel.w*.35,sel.y+sel.h*.48],[sel.x+sel.w*.65,sel.y+sel.h*.48]);
  const inputs=await processor(raw,{input_points:[points]});
  const {pred_masks,iou_scores}=await model(inputs);
  const masks=await processor.post_process_masks(pred_masks,inputs.original_sizes,inputs.reshaped_input_sizes);
  const tensor=masks[0],area=image.width*image.height,candidates=iou_scores.data.length;
  // Prefer the candidate that covers more of the user's box while penalizing
  // pixels far outside it. This avoids choosing a confident shirt-only mask.
  const padX=sel.w*.18,padY=sel.h*.18,x0=Math.max(0,sel.x-padX),x1=Math.min(image.width,sel.x+sel.w+padX),y0=Math.max(0,sel.y-padY),y1=Math.min(image.height,sel.y+sel.h+padY);
  let best=0,bestScore=-Infinity;
  for(let c=0;c<candidates;c++){let inside=0,outside=0,offset=c*area;for(let y=0;y<image.height;y++)for(let x=0;x<image.width;x++)if(tensor.data[offset+y*image.width+x]){if(x>=x0&&x<=x1&&y>=y0&&y<=y1)inside++;else outside++}const score=inside-outside*.65;if(score>bestScore){bestScore=score;best=c}}
  const offset=best*area,mask=new Uint8Array(area);
  for(let i=0;i<area;i++)mask[i]=tensor.data[offset+i]?255:0;
  return repairMask(mask,image.width,image.height);
}
function repairMask(source,w,h){
  // Small closing pass: join tiny cracks without swelling the silhouette.
  let dilated=new Uint8Array(source.length),closed=new Uint8Array(source.length);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let on=0;for(let dy=-1;dy<=1&&!on;dy++)for(let dx=-1;dx<=1;dx++){const nx=x+dx,ny=y+dy;if(nx>=0&&nx<w&&ny>=0&&ny<h&&source[ny*w+nx]){on=255;break}}dilated[y*w+x]=on}
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let on=255;for(let dy=-1;dy<=1&&on;dy++)for(let dx=-1;dx<=1;dx++){const nx=x+dx,ny=y+dy;if(nx<0||nx>=w||ny<0||ny>=h||!dilated[ny*w+nx]){on=0;break}}closed[y*w+x]=on}
  // Flood-fill background from the canvas edges. Any remaining zero island is
  // enclosed by the character silhouette, so it is an unwanted internal hole.
  const outside=new Uint8Array(source.length),queue=new Int32Array(source.length);let head=0,tail=0;
  const add=i=>{if(!closed[i]&&!outside[i]){outside[i]=1;queue[tail++]=i}};
  for(let x=0;x<w;x++){add(x);add((h-1)*w+x)}for(let y=0;y<h;y++){add(y*w);add(y*w+w-1)}
  while(head<tail){const i=queue[head++],x=i%w,y=(i/w)|0;if(x)add(i-1);if(x<w-1)add(i+1);if(y)add(i-w);if(y<h-1)add(i+w)}
  for(let i=0;i<closed.length;i++)if(!closed[i]&&!outside[i])closed[i]=255;
  return closed;
}
function trackedBox(mask,w,h,previous){
  let x0=w,y0=h,x1=0,y1=0,count=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]){x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y);count++}
  if(count<50)return previous;const px=(x1-x0)*.12,py=(y1-y0)*.12;
  return{x:Math.max(0,x0-px),y:Math.max(0,y0-py),w:Math.min(w,x1+px)-Math.max(0,x0-px),h:Math.min(h,y1+py)-Math.max(0,y0-py)};
}
async function cutout(image,sel){const mask=await inferMask(image,sel),d=image.data;for(let i=0;i<mask.length;i++)d[i*4+3]=mask[i];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.putImageData(image,0,0);return trackedBox(mask,image.width,image.height,sel)}

$('#previewButton').onclick=async()=>{
  if(busy)return;busy=true;toggle(true);$('#hint').textContent='人物の輪郭を解析中…';await new Promise(requestAnimationFrame);
  try{const img=new ImageData(new Uint8ClampedArray(frameImage.data),frameImage.width,frameImage.height);await cutout(img,selection);$('#hint').textContent='プレビュー中。人物以外が透明になっています'}
  catch(e){console.error(e);$('#hint').textContent=`輪郭解析に失敗しました: ${e.message||e}`}finally{busy=false;toggle(false)}
};
async function processFrames(onFrame){
  if(busy)return;busy=true;toggle(true);const start=video.currentTime,dur=Math.min(+controls.duration.value,video.duration-start),fps=+controls.fps.value,total=Math.max(1,Math.floor(dur*fps));let tracked={...selection};
  try{for(let n=0;n<total;n++){await seek(start+n/fps);const img=ctx.getImageData(0,0,canvas.width,canvas.height);tracked=await cutout(img,tracked);$('#hint').textContent=`輪郭AIで処理中 ${n+1} / ${total}`;await onFrame()}}
  finally{busy=false;toggle(false);$('#hint').textContent='対象キャラをドラッグで囲んでください'}
}
function toggle(v){document.querySelectorAll('button').forEach(b=>{if(b.id!=='changeButton')b.disabled=v})}
$('#exportButton').onclick=async()=>{const stream=canvas.captureStream(+controls.fps.value),chunks=[],mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm',rec=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:5_000_000});rec.ondataavailable=e=>chunks.push(e.data);rec.start();await processFrames(async()=>new Promise(r=>setTimeout(r,1000/+controls.fps.value)));rec.stop();await once(rec,'stop');showResult(new Blob(chunks,{type:mime}),'webm','背景透過対応のWebMです。対応ソフトで重ねて使えます。')};
function showResult(blob,ext,note){const url=URL.createObjectURL(blob),link=$('#downloadLink');$('#result').classList.remove('hidden');$('#resultNote').textContent=note;link.href=url;link.download=`anime-cutout.${ext}`;link.textContent=`${ext.toUpperCase()}を保存`;$('#resultVideo').src=url;$('#result').scrollIntoView({behavior:'smooth'})}
for(const [id,suffix] of [['duration','秒'],['softness','px']])controls[id].oninput=()=>$('#'+id+'Value').textContent=controls[id].value+suffix;

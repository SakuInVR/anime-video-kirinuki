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
  const raw=RawImage.fromCanvas(temp),box=[sel.x,sel.y,sel.x+sel.w,sel.y+sel.h];
  const inputs=await processor(raw,{input_boxes:[[box]]});
  const {pred_masks,iou_scores}=await model(inputs);
  const masks=await processor.post_process_masks(pred_masks,inputs.original_sizes,inputs.reshaped_input_sizes);
  let best=0;for(let i=1;i<iou_scores.data.length;i++)if(iou_scores.data[i]>iou_scores.data[best])best=i;
  const tensor=masks[0],area=image.width*image.height,offset=best*area,mask=new Uint8Array(area);
  for(let i=0;i<area;i++)mask[i]=tensor.data[offset+i]?255:0;
  return mask;
}
function trackedBox(mask,w,h,previous){
  let x0=w,y0=h,x1=0,y1=0,count=0;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]){x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y);count++}
  if(count<50)return previous;const px=(x1-x0)*.12,py=(y1-y0)*.12;
  return{x:Math.max(0,x0-px),y:Math.max(0,y0-py),w:Math.min(w,x1+px)-Math.max(0,x0-px),h:Math.min(h,y1+py)-Math.max(0,y0-py)};
}
async function cutout(image,sel){const mask=await inferMask(image,sel),d=image.data;for(let i=0;i<mask.length;i++)d[i*4+3]=mask[i];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.putImageData(image,0,0);return trackedBox(mask,image.width,image.height,sel)}

$('#previewButton').onclick=async()=>{
  if(busy)return;busy=true;toggle(true);$('#hint').textContent='人物の輪郭を解析中…';
  try{const img=new ImageData(new Uint8ClampedArray(frameImage.data),frameImage.width,frameImage.height);await cutout(img,selection);$('#hint').textContent='プレビュー中。人物以外が透明になっています'}
  catch(e){console.error(e);$('#hint').textContent='輪郭解析に失敗しました'}finally{busy=false;toggle(false)}
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

const $=s=>document.querySelector(s),video=$('#video'),canvas=$('#canvas'),maskCanvas=$('#maskCanvas');
const ctx=canvas.getContext('2d',{willReadFrequently:true}),mctx=maskCanvas.getContext('2d');
const fileInput=$('#fileInput'),controls={fps:$('#fps'),brushSize:$('#brushSize')};
let file,busy=false,painting=false,erase=false,hasPaint=false,lastPoint=null,tracker=null,trimStart=0,trimEnd=0,renderLoop=0;
const once=(el,event)=>new Promise(r=>el.addEventListener(event,r,{once:true}));

$('#pickButton').onclick=()=>fileInput.click();$('#changeButton').onclick=()=>fileInput.click();fileInput.onchange=()=>loadFile(fileInput.files[0]);
const dz=$('#dropZone');['dragenter','dragover'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.add('drag')}));['dragleave','drop'].forEach(e=>dz.addEventListener(e,x=>{x.preventDefault();dz.classList.remove('drag')}));dz.ondrop=e=>loadFile(e.dataTransfer.files[0]);
async function loadFile(f){if(!f?.type.startsWith('video/'))return alert('動画ファイルを選んでください。');file=f;video.src=URL.createObjectURL(f);await once(video,'loadedmetadata');canvas.width=maskCanvas.width=Math.min(video.videoWidth,960);canvas.height=maskCanvas.height=Math.round(canvas.width*video.videoHeight/video.videoWidth);trimStart=0;trimEnd=video.duration;$('#fileName').textContent=f.name;$('#videoMeta').textContent=`${video.videoWidth}×${video.videoHeight} / ${video.duration.toFixed(1)}秒`;$('#dropZone').classList.add('hidden');$('#workspace').classList.remove('hidden');clearMask();updateTrimUI();await seek(0)}
async function seek(t){const target=Math.max(0,Math.min(t,Math.max(0,video.duration-.001)));if(Math.abs(video.currentTime-target)<.0005){drawFrame();updateTimeUI();return}video.currentTime=target;await once(video,'seeked');drawFrame();updateTimeUI()}
function drawFrame(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(video,0,0,canvas.width,canvas.height)}
$('#timeline').oninput=()=>{video.pause();updatePlayButton();seek(video.duration*$('#timeline').value/1000)};
function point(e){const r=maskCanvas.getBoundingClientRect();return{x:(e.clientX-r.left)*maskCanvas.width/r.width,y:(e.clientY-r.top)*maskCanvas.height/r.height}}
function stroke(a,b){mctx.save();mctx.lineCap='round';mctx.lineJoin='round';mctx.lineWidth=+controls.brushSize.value;mctx.globalCompositeOperation=erase?'destination-out':'source-over';mctx.strokeStyle='rgba(255,92,138,.62)';mctx.beginPath();mctx.moveTo(a.x,a.y);mctx.lineTo(b.x,b.y);mctx.stroke();mctx.restore();hasPaint=true;$('#previewButton').disabled=$('#exportButton').disabled=false}
maskCanvas.onpointerdown=e=>{painting=true;lastPoint=point(e);maskCanvas.setPointerCapture(e.pointerId);stroke(lastPoint,lastPoint)};
maskCanvas.onpointermove=e=>{if(!painting)return;const p=point(e);stroke(lastPoint,p);lastPoint=p};maskCanvas.onpointerup=()=>{painting=false;lastPoint=null};
function setTool(isErase){erase=isErase;$('#paintButton').classList.toggle('active',!erase);$('#eraseButton').classList.toggle('active',erase)}
$('#paintButton').onclick=()=>setTool(false);$('#eraseButton').onclick=()=>setTool(true);$('#clearButton').onclick=clearMask;
function clearMask(){mctx.clearRect(0,0,maskCanvas.width,maskCanvas.height);hasPaint=false;$('#previewButton').disabled=$('#exportButton').disabled=true;$('#hint').textContent='残したいキャラをブラシで塗ってください'}
controls.brushSize.oninput=()=>$('#brushSizeValue').textContent=controls.brushSize.value+'px';

function formatTime(seconds){const m=Math.floor(seconds/60),s=Math.floor(seconds%60),ms=Math.floor((seconds-Math.floor(seconds))*1000);return`${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`}
function updateTimeUI(){$('#timeDisplay').textContent=formatTime(video.currentTime||0);$('#timeline').value=video.duration?video.currentTime/video.duration*1000:0;$('#playhead').style.left=`${video.duration?video.currentTime/video.duration*100:0}%`}
function updateTrimUI(){if(!video.duration)return;$('#startTime').textContent=formatTime(trimStart);$('#endTime').textContent=formatTime(trimEnd);$('#rangeDuration').textContent=(trimEnd-trimStart).toFixed(2)+'秒';const left=trimStart/video.duration*100,right=trimEnd/video.duration*100;$('#trimRange').style.left=left+'%';$('#trimRange').style.width=(right-left)+'%';updateTimeUI()}
function updatePlayButton(){$('#playButton').textContent=video.paused?'▶ 再生':'Ⅱ 停止'}
function animateVideo(){if(video.paused)return;drawFrame();updateTimeUI();if(video.currentTime>=trimEnd){video.pause();seek(trimEnd);updatePlayButton();return}renderLoop=requestAnimationFrame(animateVideo)}
$('#playButton').onclick=async()=>{if(video.paused){if(video.currentTime>=trimEnd-.001)await seek(trimStart);await video.play();updatePlayButton();cancelAnimationFrame(renderLoop);animateVideo()}else{video.pause();updatePlayButton()}};
$('#prevFrameButton').onclick=()=>{video.pause();updatePlayButton();seek(video.currentTime-1/(+controls.fps.value))};
$('#nextFrameButton').onclick=()=>{video.pause();updatePlayButton();seek(video.currentTime+1/(+controls.fps.value))};
$('#startJumpButton').onclick=()=>seek(trimStart);$('#endJumpButton').onclick=()=>seek(trimEnd);
$('#setStartButton').onclick=()=>{trimStart=Math.min(video.currentTime,trimEnd-.001);updateTrimUI()};
$('#setEndButton').onclick=()=>{trimEnd=Math.max(video.currentTime,trimStart+.001);updateTrimUI()};

$('#previewButton').onclick=()=>{if(!hasPaint)return;const image=ctx.getImageData(0,0,canvas.width,canvas.height),mask=mctx.getImageData(0,0,maskCanvas.width,maskCanvas.height).data;for(let i=0;i<image.width*image.height;i++)image.data[i*4+3]=mask[i*4+3]?255:0;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.putImageData(image,0,0);maskCanvas.style.opacity='0';$('#hint').textContent='初期マスクのプレビュー。塗り直すにはタイムラインを動かしてください'};
$('#timeline').addEventListener('input',()=>maskCanvas.style.opacity='1');
function status(s){const el=$('#modelStatus');el.textContent=s;if(s.includes('準備完了'))el.className='model-note ready'}
function toggle(v){busy=v;document.querySelectorAll('button').forEach(b=>{if(b.id!=='changeButton')b.disabled=v});maskCanvas.style.pointerEvents=v?'none':'auto'}
$('#exportButton').onclick=async()=>{
  if(busy||!hasPaint)return;toggle(true);maskCanvas.style.opacity='0';tracker=new CutieTracker(status);
  video.pause();updatePlayButton();const start=trimStart,dur=trimEnd-trimStart,fps=+controls.fps.value,total=Math.max(1,Math.floor(dur*fps));
  try{
    await tracker.load();tracker.reset();const stream=canvas.captureStream(fps),chunks=[],mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm',rec=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:5_000_000});rec.ondataavailable=e=>chunks.push(e.data);rec.start();
    for(let n=0;n<total;n++){await seek(start+n/fps);const raw=ctx.getImageData(0,0,canvas.width,canvas.height),result=await tracker.step(canvas,n===0?maskCanvas:null);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.putImageData(tracker.applyMask(raw,result),0,0);$('#hint').textContent=`Cutieで追跡中 ${n+1} / ${total}`;await new Promise(r=>setTimeout(r,1000/fps))}
    rec.stop();await once(rec,'stop');showResult(new Blob(chunks,{type:mime}));
  }catch(e){console.error(e);status('エラー: '+(e.message||e));$('#hint').textContent='追跡処理に失敗しました'}finally{toggle(false);maskCanvas.style.opacity='1'}
};
function showResult(blob){const url=URL.createObjectURL(blob),link=$('#downloadLink');$('#result').classList.remove('hidden');$('#resultNote').textContent='Cutieがブラシマスクを時間方向に追跡した背景透過WebMです。';link.href=url;link.download='anime-cutout.webm';link.textContent='WEBMを保存';$('#resultVideo').src=url;$('#result').scrollIntoView({behavior:'smooth'})}

class CutieTracker {
  static W=480;static H=272;static SW=30;static SH=17;static SLOTS=6;
  constructor(onStatus=()=>{}){this.onStatus=onStatus;this.reset()}
  reset(){this.sensory=this.zeros([1,1,256,17,30]);this.lastMask=null;this.objectMemory=null;this.permanent=null;this.working=[];this.frameIndex=0;this.lastMemory=-100000}
  zeros(dims){return new ort.Tensor('float32',new Float32Array(dims.reduce((a,b)=>a*b,1)),dims)}
  async load(){
    if(this.keySession)return;
    const root='models/';
    const files=[['keySession','cutie-encode-key-480x272.onnx'],['valueSession','cutie-encode-value-480x272.onnx'],['readSession','cutie-memory-readout-floatmask-valid-480x272-m6-topk30-opencv.onnx'],['decodeSession','cutie-decode-480x272.onnx']];
    let provider=navigator.gpu?'WebGPU':'WASM';
    for(let i=0;i<files.length;i++){this.onStatus(`Cutieモデルを読み込み中 ${i+1} / ${files.length}`);try{this[files[i][0]]=await ort.InferenceSession.create(root+files[i][1],{executionProviders:navigator.gpu?['webgpu']:['wasm'],graphOptimizationLevel:'all'})}catch(e){if(!navigator.gpu)throw e;provider='WASM';this.onStatus(`WebGPU非対応演算を検出。WASMへ切替中 ${i+1} / ${files.length}`);this[files[i][0]]=await ort.InferenceSession.create(root+files[i][1],{executionProviders:['wasm'],graphOptimizationLevel:'all'})}}
    this.onStatus(`Cutie準備完了（${provider}）`);
  }
  transform(width,height){const scale=Math.min(CutieTracker.W/width,CutieTracker.H/height),w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));return{width,height,w,h,x:Math.floor((CutieTracker.W-w)/2),y:Math.floor((CutieTracker.H-h)/2)}}
  imageTensor(source){
    const t=this.transform(source.width,source.height),c=document.createElement('canvas');c.width=CutieTracker.W;c.height=CutieTracker.H;const x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,c.width,c.height);x.drawImage(source,t.x,t.y,t.w,t.h);const p=x.getImageData(0,0,c.width,c.height).data,n=c.width*c.height,out=new Float32Array(n*3);for(let i=0;i<n;i++){out[i]=p[i*4]/255;out[n+i]=p[i*4+1]/255;out[n*2+i]=p[i*4+2]/255}return{tensor:new ort.Tensor('float32',out,[1,3,CutieTracker.H,CutieTracker.W]),transform:t}}
  maskTensor(source,t){const c=document.createElement('canvas');c.width=CutieTracker.W;c.height=CutieTracker.H;const x=c.getContext('2d');x.imageSmoothingEnabled=false;x.drawImage(source,t.x,t.y,t.w,t.h);const p=x.getImageData(0,0,c.width,c.height).data,out=new Float32Array(c.width*c.height);for(let i=0;i<out.length;i++)out[i]=p[i*4+3]>20?1:0;return new ort.Tensor('float32',out,[1,1,CutieTracker.H,CutieTracker.W])}
  validTensor(t){const out=new Float32Array(CutieTracker.SW*CutieTracker.SH);for(let y=0;y<CutieTracker.SH;y++)for(let x=0;x<CutieTracker.SW;x++){const px=x*16+8,py=y*16+8;if(px>=t.x&&px<t.x+t.w&&py>=t.y&&py<t.y+t.h)out[y*CutieTracker.SW+x]=1}return new ort.Tensor('float32',out,[1,1,CutieTracker.SH,CutieTracker.SW])}
  pack(kind){
    const frames=[this.permanent,...this.working].filter(Boolean).slice(0,CutieTracker.SLOTS),plane=CutieTracker.SW*CutieTracker.SH;
    if(kind==='value'){const out=new Float32Array(256*CutieTracker.SLOTS*plane);frames.forEach((f,s)=>{for(let c=0;c<256;c++)out.set(f.value.data.subarray(c*plane,(c+1)*plane),(c*CutieTracker.SLOTS+s)*plane)});return new ort.Tensor('float32',out,[1,1,256,CutieTracker.SLOTS,CutieTracker.SH,CutieTracker.SW])}
    const channels=kind==='key'?64:1,out=new Float32Array(channels*CutieTracker.SLOTS*plane);frames.forEach((f,s)=>{const src=kind==='valid'?f.valid.data:f[kind].data;for(let c=0;c<channels;c++)out.set(src.subarray(c*plane,(c+1)*plane),(c*CutieTracker.SLOTS+s)*plane)});return new ort.Tensor('float32',out,[1,channels,CutieTracker.SLOTS,CutieTracker.SH,CutieTracker.SW])
  }
  addMemory(key,shrinkage,value,valid,permanent=false){const f={key,shrinkage,value,valid};if(permanent||!this.permanent)this.permanent=f;else{this.working.push(f);while(this.working.length>5)this.working.shift()}}
  addObject(t){if(!this.objectMemory)this.objectMemory=new ort.Tensor('float32',new Float32Array(t.data),[1,1,1,16,257]);else for(let i=0;i<t.data.length;i++)this.objectMemory.data[i]+=t.data[i]}
  foreground(prob){const plane=CutieTracker.W*CutieTracker.H;return new ort.Tensor('float32',new Float32Array(prob.data.slice(plane,plane*2)),[1,1,CutieTracker.H,CutieTracker.W])}
  async step(frameCanvas,seedCanvas=null){
    const {tensor:image,transform}=this.imageTensor(frameCanvas),valid=this.validTensor(transform),k=await this.keySession.run({image});let foreground;
    if(seedCanvas)foreground=this.maskTensor(seedCanvas,transform);
    else{
      if(!this.permanent)throw new Error('初期マスクがありません');
      const r=await this.readSession.run({query_key:k.key,query_selection:k.selection,memory_key:this.pack('key'),memory_shrinkage:this.pack('shrinkage'),memory_value:this.pack('value'),memory_valid:this.pack('valid'),object_memory:this.objectMemory,pix_feat:k.pix_feat,sensory:this.sensory,last_mask:this.lastMask});
      const d=await this.decodeSession.run({f8:k.f8,f4:k.f4,memory_readout:r.memory_readout,sensory:this.sensory});this.sensory=d.new_sensory;foreground=this.foreground(d.prob);
    }
    if(seedCanvas||this.frameIndex-this.lastMemory>=5){const v=await this.valueSession.run({image,pix_feat:k.pix_feat,sensory:this.sensory,mask:foreground});this.sensory=v.new_sensory;this.addObject(v.object_memory);this.addMemory(k.key,k.shrinkage,v.mask_value,valid,!!seedCanvas);this.lastMemory=this.frameIndex}
    this.lastMask=foreground;this.frameIndex++;return{mask:foreground,transform};
  }
  applyMask(imageData,result){const {mask,transform:t}=result,d=imageData.data,w=imageData.width,h=imageData.height;for(let y=0;y<h;y++)for(let x=0;x<w;x++){const mx=Math.floor(t.x+x/w*t.w),my=Math.floor(t.y+y/h*t.h),p=mask.data[my*CutieTracker.W+mx];d[(y*w+x)*4+3]=p>=.5?255:0}return imageData}
}
window.CutieTracker=CutieTracker;

<script setup lang="ts">
import { computed, inject, reactive, ref, type Ref } from 'vue';
import DiffImage from '@/components/DiffImage.vue';
import axios, { type AxiosRequestConfig } from 'axios';
import { useFilenameStore } from '@/stores/filename';
import { storeToRefs } from 'pinia';

const backend_api = inject('backend_api') as Ref<string>

const filenameStore = useFilenameStore()
const { filename: _currentFname } = storeToRefs(filenameStore)
const { ChangeFilename } = filenameStore

const delay = (n: number) => new Promise(r => setTimeout(r, n * 1000))

const ipt = ref<HTMLInputElement | null>(null)
const originImgSrc = ref<{
  src: string,
  tag: 'origin' | 'noise'
}>({
  src: '',
  tag: 'origin'
})
const afterImgSrc = ref<string>("")
const veryOriginImgSrc = ref<string>("");

const r2 = ref<HTMLInputElement | null>(null)
const r4 = ref<HTMLInputElement | null>(null)
const r6 = ref<HTMLInputElement | null>(null)
const r8 = ref<HTMLInputElement | null>(null)
const iptFname = ref<string>("")

// status
const isOriginSkeletonDisplay = ref<boolean>(true)
const isAfterSkeletonDisplay = ref<boolean>(true)
const isAlert = ref<boolean>(false)
const isComplete = ref<boolean>(false)
const isDownloading = ref<boolean>(false)
const gaussDisable = ref<boolean>(false)
const filterDisable = ref<boolean>(false)
const isFilter = ref<boolean>(false)

const isError = ref<boolean>(false)
const isSuccess = ref<boolean>(false)

const errorIcon: string = "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
const successIcon: string = "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"

const diffmode = ref<boolean>(false)
const isCurtainRight = ref<boolean>(false)

// computed

const AlertIcon = ref<string>(errorIcon)

const mode = reactive({
  lower: true,
  mean: false,
  same: false,
  twave: false
})

function pathFormat(path: string, target: string) {
  // path: xxx/xxx.tif
  // target: jpg
  // return: xxx/xxx.jpg
  const p_list = path.split(".")
  p_list[-1] = target

  return p_list.reduce((a: string, b: string) => a + b)
}

async function addVeryOriginImg(filename: string) {
  if (filename === 'None') {
    veryOriginImgSrc.value = '';
    return Alert('success', 'Use None to del origin image in curtains.')
  }
  const link = await genImage(filename);
  console.log(link);

  if (originImgSrc.value.tag === 'noise' && link !== "Null") {
    veryOriginImgSrc.value = link;
    return
  }
  if (originImgSrc.value.tag === 'origin') {
    return Alert('error', 'This is already an origin image!')
  }
  return Alert('error', 'Unoknow error!')
}

function changeMode(_mode: 'lower' | 'mean' | 'same' | 'twave') {
  mode.lower = false
  mode.mean = false
  mode.same = false
  mode.twave = false

  switch (_mode) {
    case 'lower': mode.lower = true; break;
    case 'mean': mode.mean = true; break;
    case 'same': mode.same = true; break;
    case 'twave': mode.twave = true; break;
  }
}

function changeAlertMode(mode: 'error' | 'success') {
  isError.value = false;
  isSuccess.value = false;

  switch (mode) {
    case 'error': isError.value = true; AlertIcon.value = errorIcon; break;
    case 'success': isSuccess.value = true; AlertIcon.value = successIcon; break;
  }
}
function Alert(mode: 'error' | 'success', msg: string) {
  changeAlertMode(mode);
  isAlert.value = true;
  AlertMsg.value = msg;
  delay(4).then(() => isAlert.value = false)
}
// alert message
const AlertMsg = ref<string>("Error! Task failed successfully.")

// progress 
const progressVal = ref<number>(0)

//files data
type file = {
  filename: string
}
const filesData: Ref<file[]> = ref([{
  filename: 'None'
}])

async function refreshTable() {
  const req = await axios.get('/files_list');
  if (req.status != 200) return;
  const datas = req.data;
  filesData.value = [];
  ((datas as any).msg as file[]).forEach((data, idx) => {
    filesData.value[idx] = {
      filename: data as any as string
    }
  });
}

refreshTable()



const config: AxiosRequestConfig = {
  headers: {
    'Content-Type': 'multipart/form-data'
  },
  onUploadProgress(progressEvent) {
    const complete = Math.floor((progressEvent.loaded / (progressEvent.total as any) * 100 | 0))
    progressVal.value = complete;
  }
}

function upload() {
  const fdata = new FormData();
  const f = ipt.value?.files?.[0];

  if (f) {
    const format = f.name.split(".").pop()


    if (format === "tiff" || format === "TIF" || format === "tif") {
      progressVal.value = 0;
      isComplete.value = true;
      fdata.append('file', f);
      axios.post('/file_upload', fdata, config).then((request) => {
        if (request.data.msg === "success") {
          progressVal.value = 100;
          isComplete.value = false;
          Alert("success", "Your upload has been confirmed!")
        } else {
          progressVal.value = 0;
          isComplete.value = false;
          Alert("error", "Your upload failed!")
        }
        refreshTable()

        genImage(f.name).then((url: string) => {
          if (url !== "Null") {
            originImgSrc.value.src = url
            originImgSrc.value.tag = 'origin'
            isOriginSkeletonDisplay.value = false
            ChangeFilename(f.name)
          } else {
            Alert("error", "Image generate failed!")
          }
        })
      })
    } else {
      Alert('error', "The format only supports tiff!")
      return;
    }
  }
}
async function choose(filename: string) {
  if (_currentFname.value === filename) return Alert("success", "Image already there!")
  diffmode.value = false
  const url = await genImage(filename)
  if (url !== "Null") {
    originImgSrc.value.src = url
    if (filename.includes('noise')) {
      originImgSrc.value.tag = 'noise'
    } else { originImgSrc.value.tag = 'origin' }
    afterImgSrc.value = ""
    isOriginSkeletonDisplay.value = false
    ChangeFilename(filename)
  } else {
    Alert("error", "Image generate failed!")
  }
}


async function download(filename: string) {

  isDownloading.value = true;
  const req = await axios.get("/download", {
    params: {
      filename: filename
    },
    responseType: 'blob'
  })

  let a = document.createElement('a')
  a.download = filename
  a.style.display = 'none'
  let url = URL.createObjectURL(req.data)
  a.href = url
  document.body.appendChild(a)
  a.click()
  URL.revokeObjectURL(url)
  document.body.removeChild(a)

  isDownloading.value = false
}

async function del(filename: string) {
  const fd = new FormData()
  fd.append('filename', filename);

  await axios.post('/file_remove', fd);

  await refreshTable()
}

async function genImage(filename: string): Promise<string> {
  const req = await axios.get('/genimage', {
    params: {
      filename: filename
    }
  })
  if (req.data.msg.status === "success") {
    return backend_api.value + "/static/images/" + req.data.msg.output
  }
  return "Null"
}

type typeStd = 200 | 400 | 600 | 800;
async function GaussNoise(): Promise<string> {
  if (_currentFname.value === "") {
    Alert("error", "Choose an origin image!");
    return "Null"
  }

  filterDisable.value = true
  gaussDisable.value = true
  const std: typeStd = r2.value?.checked ? 200 : r4.value?.checked ? 400 : r6.value?.checked ? 600 : r8.value?.checked ? 800 : 200;

  const req = await axios.post('/noise', {
    filename: _currentFname.value,
    std: std
  })
  if (req.data.msg.status === "success") {
    refreshTable()
    isAfterSkeletonDisplay.value = false
    afterImgSrc.value = backend_api.value + "/static/images/" + req.data.msg.output
    gaussDisable.value = false;
  }
  gaussDisable.value = false;
  filterDisable.value = false;
  return "Null"
}

async function filter(): Promise<string> {
  console.log('----start filter----')
  filterDisable.value = true
  gaussDisable.value = true
  if (_currentFname.value == "") {
    Alert("error", "Choose an origin image!")
    return "Null"
  }
  // Confirm which mode is active
  const method = mode.mean ? 'mean' : mode.lower ? 'lower' : mode.same ? 'same' : 'twave';
  const req = await axios.get('/filter', {
    params: {
      filename: _currentFname.value,
      method: method
    }
  })
  if (req.data.msg.status === "success") {
    refreshTable()
    isAfterSkeletonDisplay.value = false
    afterImgSrc.value = backend_api.value + "/static/images/" + req.data.msg.output
  }
  filterDisable.value = false
  gaussDisable.value = false
  return "Null"
}

</script>

<template>
  <div class="wrapper flex flex-col space-y-5 justify-center items-center">
    <div class="header flex flex-row space-x-10">
      <form @submit.prevent="upload" enctype="multipart/form-data"
        class="relative flex flex-row space-x-3 pt-5 font-[Simsun] ">
        <input accept=".TIF,.tif,.tiff,.TIFF" class=" file-input file-input-bordered w-80 " type="file" ref="ipt" />
        <progress class="cursor-progress progress progress-success w-56 absolute left-20 bottom-0" :value=progressVal
          max="100"></progress>
        <span v-if="isComplete" class="loading loading-spinner text-success absolute right-20 top-8"></span>
        <button type="submit" class="btn btn-outline">上传</button>
      </form>

      <label class="flex flex-col space-y-2 items-center justify-center"
        v-if="originImgSrc.src !== '' && afterImgSrc !== ''">
        <span class="font-[Simsun] font-bold">卷帘</span>
        <input type="checkbox" class="toggle" v-model="diffmode" alt="Switch Mode" />
      </label>
      <label class="flex flex-col space-y-2 items-center justify-center" v-if="veryOriginImgSrc !== '' && diffmode">
        <span class="font-[Simsun] font-bold">左/右</span>
        <input type="checkbox" class="toggle" v-model="isCurtainRight" alt="Switch Mode" />
      </label>
      <form v-if="diffmode" @submit.prevent="addVeryOriginImg(iptFname)" class="pt-5 font-[Simsun] pl-5">
        <label class="input input-bordered flex items-center gap-2">
          <input type="text" class="grow" placeholder="Search" v-model="iptFname" />
          <button type="submit" class="kbd kbd-sm">Y</button>
        </label>
      </form>
    </div>

    <div v-if="!diffmode" class="file flex flex-row items-center space-x-2 relative">
      <div class="origin relative h-[30rem] w-[30rem] ">
        <h2 class="text-center">处理前</h2>
        <div class="content border border-black absolute border-dashed w-full h-full">
          <div v-if="isOriginSkeletonDisplay" class="bg-base-200 skeleton h-full w-full">
            <span class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 ">原图</span>
          </div>
          <img alt="" class="cursor-pointer" srcset="" :src="originImgSrc.src">
        </div>
      </div>
      <label class="absolute left-1/2 top-5 -translate-x-1/2 flex flex-col space-y-2 items-center justify-center">
        <span class="font-[Simsun] font-bold">加躁/降噪</span>
        <input type="checkbox" class="toggle " v-model="isFilter" alt="Switch Mode" />
      </label>
      <div class="panel w-28 font-[Simsun] justify-center items-center flex flex-col space-y-5">
        <div class="filter flex flex-col items-center justify-center" v-if="isFilter">
          <ul class="w-32 menu">
            <li class="flex items-center justify-center"><a :class="{ active: mode.lower }"
                @click="changeMode('lower')">低通滤波</a></li>
            <li class="flex items-center justify-center"><a :class="{ active: mode.mean }"
                @click="changeMode('mean')">均值滤波</a></li>
            <li class="flex items-center justify-center"><a :class="{ active: mode.same }"
                @click="changeMode('same')">同态滤波</a></li>
            <li class="flex items-center justify-center"><a :class="{ active: mode.twave }"
                @click="changeMode('twave')">模糊集滤波</a></li>
          </ul>
          <button class="btn btn-outline btn-md" @click="filter()" :disabled="filterDisable">->处理-></button>

        </div>
        <div v-else class="gauss flex flex-col justify-center items-center">
          <button class="btn btn-outline btn-xs text-center mt-2" :disabled="gaussDisable"
            @click="GaussNoise">->高斯噪声-></button>
          <div class="params grid grid-cols-2 grid-rows-2">
            <div class="form-control">
              <label class="label cursor-pointer">
                <input type="radio" name="radio-1" class="radio  radio-xs" ref="r2" checked="true" />
                <span class="label-text pl-1">200</span>
              </label>
            </div>
            <div class="form-control">
              <label class="label cursor-pointer">
                <input type="radio" name="radio-1" class="radio  radio-xs" ref="r4" />
                <span class="label-text">400</span>
              </label>
            </div>
            <div class="form-control">
              <label class="label cursor-pointer">
                <input type="radio" name="radio-1" class="radio  radio-xs" ref="r6" />
                <span class="label-text">600</span>
              </label>
            </div>
            <div class="form-control">
              <label class="label cursor-pointer">
                <input type="radio" name="radio-1" class="radio  radio-xs" ref="r8" />
                <span class="label-text">800</span>
              </label>
            </div>
          </div>
        </div>


      </div>

      <div class="after relative h-[30rem] w-[30rem] ">
        <h2 class="text-center">处理后</h2>
        <div class="content border border-black absolute border-dashed w-full h-full">
          <div v-if="isAfterSkeletonDisplay" class=" bg-base-200 skeleton h-full w-full">
            <span class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">处理后</span>
          </div>
          <img :src="afterImgSrc" class="cursor-pointer">
        </div>
      </div>
    </div>
    <DiffImage v-else :src-noise="originImgSrc.src" :src-filter="afterImgSrc" :src-origin="veryOriginImgSrc"
      :is-right="isCurtainRight" src-o-o="http://127.0.0.1:5000/static/images/L8_20190329_Clip._mean_filter.jpg"
      class="h-[30rem] w-[30rem]" />
    <Transition name="bounce">
      <div v-if="isAlert" role="alert" class="alert  w-11/12"
        :class="{ 'alert-error': isError, 'alert-success': isSuccess }">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d=AlertIcon />
        </svg>
        <span>{{ AlertMsg }}</span>
      </div>
    </Transition>

    <!-- files -->
    <div class="overflow-y-auto max-h-[30rem] pt-10">
      <table class="table table-pin-rows ">
        <!-- head -->
        <thead>
          <tr class="font-[Simsun] text-xl text-center ">
            <th>序号</th>
            <th class="w-48">文件名</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <!-- row 1 -->
          <tr v-for="(item, index) in filesData" :key="index" class="font-[Times] text-xl">
            <th>{{ index }}</th>
            <td>{{ item.filename }}</td>
            <th class="flex flex-row space-x-3">
              <button class="btn btn-warning btn-xs" @click="del(item.filename)">Delete</button>
              <button class="btn btn-accent btn-xs" @click="choose(item.filename)">Choose</button>
              <button class="btn btn-accent btn-xs" @click="download(item.filename)">Download</button>
            </th>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="mask scale-75 w-full h-full bg-gray-100 absolute opacity-80 flex flex-col items-center justify-center"
      v-if="isDownloading">
      <h1 class="font-semibold text-2xl">Downloading...</h1>
      <div class="load">
        <span class="loading loading-bars loading-xs"></span>
        <span class="loading loading-bars loading-sm"></span>
        <span class="loading loading-bars loading-md"></span>
        <span class="loading loading-bars loading-lg"></span>
      </div>
    </div>
  </div>

</template>

<style scoped>
.bounce-enter-active {
  animation: bounce-in 0.5s;
}

.bounce-leave-active {
  animation: bounce-in 0.5s reverse;
}

@keyframes bounce-in {
  0% {
    transform: scale(0);
  }

  50% {
    transform: scale(1.1);
  }

  100% {
    transform: scale(1);
  }
}
</style>

import './assets/main.css'

// import { GPU } from 'gpu.js';

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

const app = createApp(App)
// const gpu = new GPU()

// app.provide('gpu', gpu)

app.use(createPinia())
app.use(router)

app.mount('#app')

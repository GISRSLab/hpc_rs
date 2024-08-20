<script setup lang="ts">
import { onMounted, provide, watch } from 'vue';
import { RouterLink, RouterView } from 'vue-router'
import { useLocalStorage } from '@vueuse/core';
import axios from 'axios';

const backend_api = useLocalStorage('backend_api', import.meta.env.VITE_BACKEND_API)
provide('backend_api', backend_api)
watch(backend_api, (val: string) => {
  axios.defaults.baseURL = val;
}, { immediate: true })

</script>

<template>
  <div class="navbar bg-base-100 relative z-50">
    <div class="navbar-start ">
      <a class="btn btn-ghost text-xl font-mono glass">异构计算加速的遥感影像滤波快速处理系统</a>
    </div>
    <div class="navbar-center glass hidden rounded-md lg:flex ">
      <ul class="menu menu-horizontal px-1 font-semibold font-[Times]">
        <li><router-link to="/" class="">Home</router-link></li>

        <li><router-link to="/about" class="">About</router-link></li>
      </ul>
    </div>
    <div class="navbar-end">
      <div tabindex="0" role="button" class="btn btn-ghost btn-circle avatar">
        <div class="w-10 rounded-full">
          <img alt="Navbar component" src="/imgs/earth.jpg" />
        </div>
      </div>
    </div>
  </div>
  <div class="midst flex items-center justify-center">
    <RouterView />
  </div>
  <footer class="absolute bottom-0 footer footer-center bg-base-100 font-[Times]  text-base-content p-4">
    <aside>
      <p>Copyright © {{ new Date().getFullYear() }} - All right reserved by Zayn</p>
    </aside>
  </footer>
</template>

<style scoped></style>

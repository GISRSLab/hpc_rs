import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export const useFilenameStore = defineStore('filename', () => {
  const filename = ref<string>("") // end with tiff-like

  function ChangeFilename(n: string) {
    filename.value = n
  }

  return { filename, ChangeFilename }
})

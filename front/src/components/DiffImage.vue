<script setup lang="ts">
import { ref, computed, watch, type WatchStopHandle } from 'vue';

const props = withDefaults(defineProps<{
    isRight: boolean,
    srcNoise: string,
    srcFilter: string,
    srcOrigin?: string
}>(), {
    isRight: true,
    srcNoise: '',
    srcFilter: ''
})

const left_val = ref<number>(20)
const right_val = ref<number>(80)


watch(left_val, (newVal) => {
    if (right_val.value < newVal) {
        right_val.value = newVal
    }
})

watch(right_val, (newVal) => {
    if (left_val.value > newVal) {
        left_val.value = newVal
    }
})

watch(() => props.srcOrigin, () => {
    if (props.srcOrigin === '' || !props.srcOrigin) {
        left_val.value = 0;
    }
}, { immediate: true })

const left = computed<string>(() => {
    return left_val.value + '%';
})
const right = computed<string>(() => {
    return right_val.value + '%';
})
const LTag = computed(() => {
    return props.srcOrigin && props.srcOrigin !== '' ? 'origin' : ''
})


const thumb_color_right = computed(() => {
    return props.isRight ? 'lightblue' : 'white';
})
const thumb_color_left = computed(() => {
    return props.isRight ? 'white' : 'greenyellow';
})
</script>

<template>
    <div class="zdiff relative select-none ">
        <div
            class="info flex flex-row space-x-20 text-center absolute -bottom-5 translate-y-1/2 left-1/2 -translate-x-1/2">
            <span>{{ LTag }}</span>
            <div class="divider divider-horizontal"></div>
            <span>noise</span>
            <div class="divider divider-horizontal"></div>
            <span>filter</span>
        </div>
        <img class="absolute w-full h-full item3" alt="oo" :src="props.srcFilter" />
        <img class="absolute w-full h-full item2" alt="after" :src="props.srcNoise" />
        <img class="absolute w-full h-full item1" alt="origin" :src="props.srcOrigin"
            v-if="props.srcOrigin !== '' && props.srcOrigin" />
        <input type="range" min="0" max="100" v-model="left_val" class="range absolute w-full h-full "
            v-if="props.srcOrigin !== '' && props.srcOrigin" :class="{ 'z-50': !props.isRight }" id="left" />
        <input type="range" min="0" max="100" v-model="right_val" class="range absolute w-full h-full " id="right" />
    </div>
</template>

<style scoped>
.zdiff {
    --left: v-bind('left');
    --right: v-bind('right');
    --thumb_left: v-bind('thumb_color_left');
    --thumb_right: v-bind('thumb_color_right');
}

.item1 {
    clip-path: polygon(0% 0%, 0% 100%, var(--left) 100%, var(--left) 0%);
}

.item2 {
    clip-path: polygon(var(--left) 100%, var(--left) 0%, var(--right) 0%, var(--right) 100%);
}

.item3 {
    clip-path: polygon(var(--right) 0%, var(--right) 100%, 100% 100%, 100% 0%);
}

.control1 {
    left: var(--left);
}

.control2 {
    left: var(--right);
}

.range {
    --range-shdw: linear-gradient(to right, rgba(254, 255, 255, 1) 96%, rgba(255, 0, 4, 1) 100%);
}

::-webkit-slider-container {
    /*可以修改容器的若干样式*/
}

::-webkit-slider-runnable-track {
    /*可以修改轨道的若干样式*/
    background-color: var(--range-shdw);
}

::-webkit-slider-thumb {
    /*::-webkit-slider-thumb是代表给滑块的样式进行变更*/
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    /*//这三个是去掉滑块原有的默认样式，划重点！！*/
    -webkit-box-shadow: 0 0 2px;
    /*设置滑块的阴影*/
    /*//这几个是设置滑块的样式*/
}

#left::-webkit-slider-thumb {
    background: var(--thumb_left);
}

#right::-webkit-slider-thumb {
    background: var(--thumb_right);
}
</style>
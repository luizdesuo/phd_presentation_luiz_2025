<template>
  <Scroll />
  <footer
    v-if="$nav.currentLayout !== 'intro'"
    class="absolute bottom-2 right-2"
  >
    <br /><small
      >Luiz Desuó Neto • Untying the Gordian Knot of Bayesian Inference in
      Markov Networks • <SlideCurrentNo />/<SlidesTotal /></small
    ><br />
  </footer>
</template>

import RetrievalSystem from "./RetrievalSystem.ts";
import KMeans from "./KMeans.js";
import * as tf from '@tensorflow/tfjs-node-gpu';


export default class KMeansRetrievalSystem extends RetrievalSystem {
  clustercount: number = 1000;
  clusters: ClusterData[] = [];

  async fit(documents: string[], kmeansIt: number = 10): Promise<void> {
    await super.fit(documents);

    let kmeans: KMeans = new KMeans(this.clustercount, kmeansIt) // to prepare all documents embeddings and norms
    let [centroids, clusterAssignments]: [tf.Tensor2D, tf.Tensor1D] = kmeans.run(this.documentsEmbeddings!);

    const clusterAssignmentsArray = clusterAssignments.arraySync();
    clusterAssignments.dispose(); // free some memory

    let centroidsArray = centroids.arraySync();

    // construct a clusterData object for each cluster
    for (let centroidIndex = 0; centroidIndex < centroidsArray.length; centroidIndex++) {
      let vectorIndeces = clusterAssignmentsArray.map((x: number, index: number) => x === centroidIndex ? index : -1).filter((x: number) => x !== -1);
      let clusterVectors = vectorIndeces.map((i) => this.documentsEmbeddings[i])
      let centroid = centroids.gather([centroidIndex]).reshape([-1]) as tf.Tensor1D;
      this.clusters.push(new ClusterData(clusterVectors, centroid.div(centroid.norm()), vectorIndeces));
      centroid.dispose();
    }
    console.log("Finished fitting");
    // this.documentsNorms = [];
    this.documentsEmbeddings = [];
  }


  async query(query: string, k: number = 10, clusterSearchCount: number = 1): Promise<[documentindex: number, score: number][]> {

    const embeddings = this.clusters.map((c) => c.centroid.arraySync()) as number[][];

    let options: [number, number][] = [];
    for (let i = 0; i < embeddings.length; i += 10000) {
      let rep = tf.tensor2d(embeddings.slice(i, Math.min(i + 10000, embeddings.length)))
      let x = await this._query(query, rep, clusterSearchCount);
      options.push(...x);
      rep.dispose();
    }
    options.sort((a, b) => b[1] - a[1]);
    options = options.slice(0, k);


    let allResults : [documentindex: number, score: number][] = [];

    let indexcount= 0;
    // iterate over the closest centroid(s)
    for (let [centroidIndex, _] of options) {
      indexcount++;
      const clustervector_gpu_loaded = tf.tensor2d(this.clusters[centroidIndex].documentVectors);
      const topk = await this._query(query, clustervector_gpu_loaded,  k);
      clustervector_gpu_loaded.dispose();
      // topk has indeces of the documents in the cluster, we need to convert them to the indeces of the documents in the original list
      const realIndeces: [documentindex: number, score: number][] = topk.map((doc) => [this.clusters[centroidIndex].documentIndeces[doc[0]], doc[1]]);
      allResults = [...allResults, ...realIndeces];
    }

    console.log("Finished querying");

    // sort by score if necessary
    if (allResults.length > k) {
      allResults.sort((a, b) =>  b[1] - a[1])
      allResults = allResults.slice(0, k);
    }

    return allResults;

  }
}



// this class hold all data of a cluster
class ClusterData {
  documentVectors: number[][];
  centroid: tf.Tensor1D;
  documentIndeces: number[];

  constructor(documentVectors: number[][], centroid: tf.Tensor1D, documentIndeces: number[]) {
    this.documentVectors = documentVectors;
    this.centroid = centroid
    this.documentIndeces = documentIndeces;
  }


}
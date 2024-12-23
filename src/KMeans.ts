import * as tf from '@tensorflow/tfjs-node-gpu';
import {Rank} from "@tensorflow/tfjs-node-gpu";


export default class KMeans {

  clustercount: number = 3;
  max_iter: number = 10;

  constructor(n_clusters: number, max_iter: number = 10) {
    this.clustercount = n_clusters;
    this.max_iter = max_iter;
  }

  private _initCentroids(vectors: number[][]): tf.Tensor2D {
    let centroidnumbers: number[] = []
    for (let i = 0; i < this.clustercount; i++) {
      let rand_val = Math.floor(Math.random() * vectors.length);
      centroidnumbers.push(rand_val);
    }
    return tf.tensor2d(centroidnumbers.map((i) => vectors[i]));
  }


  private _updateCentroids(vectors: number[][], clusterAssignments: tf.Tensor1D): tf.Tensor2D {
    const newCentroids = [];
    console.log("Updating centroids");

    // iterate over all clusters, gather the vectors that belong to the cluster and calculate the mean
    for (let i = 0; i < this.clustercount; i++) {
      if (clusterAssignments.shape[0] < 10000000000000) {
        const assi = clusterAssignments.arraySync().map((x: number, index: number) => x === i ? index : -1).filter((x: number) => x !== -1)
        let new_centroid = []; for (let j = 0; j < vectors[0].length; j++) { new_centroid.push(0); }
        for (let j = 0; j < assi.length; j++) {
          for (let k = 0; k < vectors[0].length; k++) {
            new_centroid[k] += vectors[assi[j]][k];
          }
        }
        for (let j = 0; j < new_centroid.length; j++) {
          new_centroid[j] /= assi.length;
        }
        newCentroids.push(tf.tensor1d(new_centroid));

      }
      else {
        let clusterVectors: number[][] = [];
        for (let j = 0; j < clusterAssignments.shape[0]; j += 10000) {
          const tmp_dd = tf.tensor2d(vectors.slice(j, Math.min(j + 10000, vectors.length)));
          clusterVectors.push(...tmp_dd.gather(clusterAssignments.arraySync().slice(j, Math.min(j + 10000, vectors.length)).map((x: number, index: number) => x === i ? index : -1).filter((x: number) => x !== -1)).arraySync());
          tmp_dd.dispose();
        }
        const nedsdsd = tf.tensor2d(clusterVectors);
        newCentroids.push(nedsdsd.mean(0));
        nedsdsd.dispose();
      }
    }

    // stack the new centroids into a matrix
    return tf.stack(newCentroids) as tf.Tensor2D;
  }


  private _assignVectorsToCentroids(vectors: number[][], centroids: tf.Tensor2D): tf.Tensor1D {

    if (vectors.length < 10000) {
      // if the number of vectors is small, we can calculate the distances between each vector and each centroid directly
      console.log("Calculating distances directly");
      const distances = tf.norm(tf.tensor2d(vectors).expandDims(1).sub(centroids.expandDims(0)), "euclidean", 2).squeeze();
      return distances.argMin(1); // return the index of the closest centroid
    }

    let closest: number[] = [];

    const expandedCentroids = centroids.expandDims(0);

    // split the vectors in batches because they are too large to calculate all at once
    for (let i = 0; i < vectors.length; i += 1000) {
      console.log(`Processing distances for ${i} vectors`);
      const minmin = Math.min(i + 1000, vectors.length); // so I dont go out of bounds
      const expandedVectors = tf.tensor2d(vectors.slice(i, minmin)).expandDims(1); // expand the vectors to be able to subtract the centroids
      const subbed = expandedVectors.sub(expandedCentroids); // subtract the centroids from the vectors
      expandedVectors.dispose();
      const distances = tf.norm(subbed, "euclidean", 2).squeeze(); // calculate the distances
      subbed.dispose();
      closest.push(...(distances.argMin(1) as tf.Tensor1D).arraySync());
      distances.dispose();
    }

    expandedCentroids.dispose();
    return tf.tensor1d(closest, "int32");
  }


  public run(vectors: number[][]) : [centroids: tf.Tensor2D, centroidAssignments: tf.Tensor1D] {
    // select centroids by choosing random vectors from the input
    let centroids = this._initCentroids(vectors);

    console.log("Initialized centroids");

    // assigne each vector to the closest centroid
    let clusterAssignments = this._assignVectorsToCentroids(vectors, centroids);

    // takes the norm of the counts of elements in the cluster assignments. This will be used as a stopping criterion
    // @ts-ignore // I need to add this line so typescript doesnt freak out for givng null as a parameter
    let clusterAssignmentsCountsNorm = tf.norm(tf.bincount(clusterAssignments, [], this.clustercount -1), "euclidean", null);

    for (let i = 0; i < this.max_iter; i++) {
      // update the centroids
      centroids = this._updateCentroids(vectors, clusterAssignments);

      // reassign the vectors to the closest centroid
      clusterAssignments = this._assignVectorsToCentroids(vectors, centroids);


      // check if the cluster assignments have changed enough to continue the iteration
      // @ts-ignore
      const newClusterAssignmentsCountsNorm = tf.norm(tf.bincount(clusterAssignments, [], this.clustercount -1), "euclidean", null); // yes both are scalars
      let divisionOverCountNorms = newClusterAssignmentsCountsNorm.div(clusterAssignmentsCountsNorm) as tf.Tensor<Rank.R0>  // weird method because both are scalars disguised as tensors
      if (divisionOverCountNorms.dataSync()[0] < 1.001 && divisionOverCountNorms.dataSync()[0] > 0.999) {
        break;
      }
      clusterAssignmentsCountsNorm = newClusterAssignmentsCountsNorm;
    }
    console.log("Finished clustering");

    return [centroids, clusterAssignments];
  }


}
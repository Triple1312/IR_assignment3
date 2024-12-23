import * as tf from '@tensorflow/tfjs-node-gpu';
import {FeatureExtractionPipeline, pipeline, Tensor} from "@huggingface/transformers";
import {tensor1d} from "@tensorflow/tfjs-node-gpu";

export default class RetrievalSystem {
  static model_name: string = "sentence-transformers/all-MiniLM-L6-v2";
  static merge_mode: string = "mean";
  featureExtractor: FeatureExtractionPipeline | null = null;

  documentsEmbeddings: number[][] = [];

  async embedDocument(document: string): Promise<number[]> {
    return (await this.embedDocuments([document]))[0];
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    if (this.featureExtractor === null) {
      throw new Error("Model not trained");
    }

    // turning the text into a tensor of token vectors
    const embeddings = await this.featureExtractor(documents);
    const embeddingsList: number[][][] = embeddings.tolist();

    embeddings.dispose(); // to free the space used by transformers.js

    // merging the token vectors into one document vector
    switch (RetrievalSystem2.merge_mode){
      case 'mean':
        return embeddingsList.map((embedding) => {
          let resultvector = embedding[0];
          for (let i = 1; i < embedding.length; i++ ) {
            for (let j = 0; j < embedding[i].length; j++) {
              resultvector[j] += embedding[i][j];
            }
          }
          for (let i = 0; i < resultvector.length; i++) {
            resultvector[i] /= embedding.length;
          }
          return resultvector;
        });
      case 'max':
        return embeddingsList.map((embedding) => embedding.reduce((acc, val) => acc.map((v, i) => Math.max(v, val[i]))));
      default:
          throw 'Invalid type'
    }

  }


  /// Fit the retrieval system to a list of documents
  async fit(documents: string[]) : Promise<void> {
    if (this.featureExtractor === null) {
      this.featureExtractor = await pipeline('feature-extraction', RetrievalSystem2.model_name, {device: 'cpu'});
    }

    let tmp_docs = []

    const batchSize = 100;
    for (let documentIndex: number = 0; documentIndex < documents.length; documentIndex += batchSize) {
      let x = await this.embedDocuments(documents.slice(documentIndex, Math.min(documentIndex + batchSize, documents.length)));
      tmp_docs.push(...x);
      console.log(`Processed ${documentIndex + batchSize} documents`);
    }


    for (let i = 0; i < tmp_docs.length; i++) {
      let eek = tensor1d(tmp_docs[i]);
      // normalizing the document embeddings so cosine sim is easier
      this.documentsEmbeddings.push(eek.div(eek.norm()).arraySync() as number[]);
      eek.dispose();
    }
  }


  async _query(query_string: string, documentsEmbeddings: tf.Tensor2D, k: number = 10): Promise<[documentindex: number, score: number][]> {
    const queryTensor = tf.tensor1d(await this.embedDocument(query_string));

    // normalizing the query so cosine sim is easier
    const query_norm = queryTensor.norm();
    const normalized_query = queryTensor.div(query_norm);
    queryTensor.dispose();
    query_norm.dispose();

    // calculate the cosine similarity between the query and all documents
    // the reshapes are so the tensors are the right shape for the matrix multiplication
    const cosinesims = documentsEmbeddings!.matMul(normalized_query.reshape([-1,1])).reshape([-1]);

    // shaping all the results into the right format
    const {values, indices} = cosinesims.topk(cosinesims.shape[0] < k? cosinesims.shape[0] : k);
    let valuesArray: number[] = await values.array() as number[];
    let indicesArray: number[] = await indices.array() as number[];
    let results: [number, number][] = [];
    for (let i = 0; i < k; i++) {
      results.push([indicesArray[i], valuesArray[i]]);
    }


    // dispose of all Tensors since the js garbage collector doesnt free graphics memory
    queryTensor.dispose();
    cosinesims.dispose();
    values.dispose();
    indices.dispose();
    query_norm.dispose();
    normalized_query.dispose();

    return results;
  }


  /// @return: a list of tuples with the document index and the score
  async query(query_string: string, k: number = 10): Promise<[documentindex: number, score: number][]> {

    // because otherwise I cant load it into my gpu with tensorflow
    if (this.documentsEmbeddings.length < 10000) {
      return await this._query(query_string, tf.tensor2d(this.documentsEmbeddings),  k);
    }

    let options: [number, number][] = [];
    for (let i = 0; i < this.documentsEmbeddings.length; i += 10000) {
      const embeddings = tf.tensor2d(this.documentsEmbeddings.slice(i, Math.min(i + 10000, this.documentsEmbeddings.length)))
      let x = await this._query(query_string, embeddings, k);
      embeddings.dispose();
      options.push(...x);
    }
    options.sort((a, b) => b[1] - a[1]);
    return options.slice(0, k);

  }



}
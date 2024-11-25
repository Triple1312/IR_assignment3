import { FeatureExtractionPipeline,  pipeline, Tensor } from "npm:@huggingface/transformers";


export default class RetrievalSystem {
  static model_name: string = "sentence-transformers/all-MiniLM-L6-v2";
  static merge_mode: string = "mean";
  documentEmbeddings :number[][] = [];
  documentsNorms: number[] = [];
  featureExtractor: FeatureExtractionPipeline | null = null;

  async fit(documents: string[]) : Promise<void> {
    if (this.featureExtractor === null) {
      this.featureExtractor = await pipeline('feature-extraction', RetrievalSystem.model_name);
    }
    let document_index = 0;
    for (const document  of documents) {
      document_index++;
      const embedding: Tensor = await this.featureExtractor(document);
      const embeddings_list: number[][] = embedding.map((t: number[][]) => t).tolist()[0];
      let eek: number[] = [];
      for (let i = 0; i < 384; i++) eek.push(0);
      if (RetrievalSystem.merge_mode == "mean" ) {
        for (const vec of embeddings_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] += vec[i];
          }
        }
        for (let i = 0; i < 384; i++) {
          eek[i] /= embeddings_list.length;
        }
      }
      else if (RetrievalSystem.merge_mode == "max") {
        for (const vec of embeddings_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] = vec[i] > eek[i] ? vec[i] : eek[i];
          }
        }
      }
      else if (RetrievalSystem.merge_mode == "max_abs") {
        for (const vec of embeddings_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] = Math.abs(vec[i]) > Math.abs(eek[i]) ? vec[i] : eek[i];
          }
        }
      }
      else {
        throw new Error("Invalid mode");
      }
      this.documentEmbeddings.push(eek);

      // calculate the norm of the document embedding for faster queries with cosine similarity
      let tmp_norm: number = 0;
      for (let i = 0; i < 384; i++) {
        tmp_norm += eek[i] * eek[i];
      }
      this.documentsNorms.push(Math.sqrt(tmp_norm));
      if (document_index % 100 == 0) {
        console.log(`Processed ${document_index} documents`);
      }
    }
  }

  /// Query the documents and return the top k most similar documents
  async query(query_string: string, k: number = 10): Promise<number[]> {
    let scores: [number, number][] = [];
    if (this.featureExtractor === null) {
      throw new Error("Model not trained");
    }
    const query_embedding: Tensor = await this.featureExtractor(query_string);
    const query_embedding_list: number[][] = query_embedding.map((t: number[][]) => t).tolist()[0];
    let eek: number[] = [];
    for (let i = 0; i < 384; i++) eek.push(0);

    /// because branching is slow in javascript
    if (RetrievalSystem.merge_mode == "mean" ) {
      for (const vec of query_embedding_list) {
        for (let i = 0; i < 384; i++) {
          eek[i] += vec[i];
        }
      }
      for (let i = 0; i < 384; i++) {
        eek[i] /= query_embedding_list.length;
      }
    }
    else if (RetrievalSystem.merge_mode == "max") {
      for (const vec of query_embedding_list) {
        for (let i = 0; i < 384; i++) {
          eek[i] = vec[i] > eek[i] ? vec[i] : eek[i];
        }
      }
    }
    else if (RetrievalSystem.merge_mode == "max_abs") {
      for (const vec of query_embedding_list) {
        for (let i = 0; i < 384; i++) {
          eek[i] = Math.abs(vec[i]) > Math.abs(eek[i]) ? vec[i] : eek[i];
        }
      }
    }
    else {
      throw new Error("Invalid mode");
    }

    let query_norm: number = 0;
    for (let i = 0; i < 384; i++) {
      query_norm += eek[i] * eek[i];
    }
    query_norm = Math.sqrt(query_norm);

    for (let i = 0; i < this.documentEmbeddings.length; i++) {
      let dot_product: number = 0;
      for (let j = 0; j < 384; j++) {
        dot_product += eek[j] * this.documentEmbeddings[i][j];
      }

      if (i % (8* k)) { // to minimize the number of sorts and space
        scores.sort((a, b) => b[1] - a[1]);
        scores = scores.slice(0, k);
      }

      scores.push([i, dot_product / (query_norm * this.documentsNorms[i])]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    if (scores.length < k) {
      k = scores.length;
    }
    return scores.map((a) => a[0]).slice(0, k);
  }

  async queries(queries: string[], k: number = 10): Promise<number[][]> {
    let results: number[][] = [];
    let query_index =0;
    // yes a lot of code duplication but it's faster because of js branch performance
    for (const query_string of queries) {
      query_index++;
      let scores: [number, number][] = [];
      if (this.featureExtractor === null) {
        throw new Error("Model not trained");
      }
      const query_embedding: Tensor = await this.featureExtractor(query_string);
      const query_embedding_list: number[][] = query_embedding.map((t: number[][]) => t).tolist()[0];
      let eek: number[] = [];
      for (let i = 0; i < 384; i++) eek.push(0);

      /// because branching is slow in javascript
      if (RetrievalSystem.merge_mode == "mean" ) {
        for (const vec of query_embedding_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] += vec[i];
          }
        }
        for (let i = 0; i < 384; i++) {
          eek[i] /= query_embedding_list.length;
        }
      }
      else if (RetrievalSystem.merge_mode == "max") {
        for (const vec of query_embedding_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] = vec[i] > eek[i] ? vec[i] : eek[i];
          }
        }
      }
      else if (RetrievalSystem.merge_mode == "max_abs") {
        for (const vec of query_embedding_list) {
          for (let i = 0; i < 384; i++) {
            eek[i] = Math.abs(vec[i]) > Math.abs(eek[i]) ? vec[i] : eek[i];
          }
        }
      }
      else {
        throw new Error("Invalid mode");
      }

      let query_norm: number = 0;
      for (let i = 0; i < 384; i++) {
        query_norm += eek[i] * eek[i];
      }
      query_norm = Math.sqrt(query_norm);

      for (let i = 0; i < this.documentEmbeddings.length; i++) {
        let dot_product: number = 0;
        for (let j = 0; j < 384; j++) {
          dot_product += eek[j] * this.documentEmbeddings[i][j];
        }

        if (i % (8* k)) { // to minimize the number of sorts and space
          scores.sort((a, b) => b[1] - a[1]);
          scores = scores.slice(0, k);
        }

        scores.push([i, dot_product / (query_norm * this.documentsNorms[i])]);
      }
      scores.sort((a, b) => b[1] - a[1]);
      if (scores.length < k) {
        k = scores.length;
      }
      results.push(scores.map((a) => a[0]).slice(0, k));
      if (query_index % 100 == 0) {
        console.log(`Processed ${query_index} queries`);
      }

    }
    return results;
  }

}
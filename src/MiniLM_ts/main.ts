import { pipeline } from "npm:@huggingface/transformers";
import RetrievalSystem from './RetrievalSystem.ts';
import DataFrame from "./DataFrame.ts";

let queries: DataFrame = await DataFrame.fromCSV("../data/docs_small/dev_small_queries.csv");
console.log("loaded queries");
let fileNames: string[] = []
let fileContents: string[] = [];

// load all documents
for await (const dirEntry of Deno.readDir("../data/docs_small/full_docs_small")) {
  if (dirEntry.isFile && dirEntry.name.endsWith(".txt")) {
    fileNames.push(dirEntry.name);
    fileContents.push(await Deno.readTextFile("../data/docs_small/full_docs_small/" + dirEntry.name));
  }
}
console.log("loaded files");
let retrievalSystem: RetrievalSystem = new RetrievalSystem();
await retrievalSystem.fit(fileContents);
let x = await retrievalSystem.queries(queries.data.map((row: string[]) => row[1]), 10);
let file_string: string = "Query_number,doc_number\n";
for (let i = 0; i < queries.data.length; i++) {
  for (let j = 0; j < x[i].length; j++) {
    file_string += `${queries.data[i][0]},${fileNames[x[i][j]].split('_')[1].split('.')[0].toString()}\n`;
  }
}
file_string = file_string.slice(0, -1);
const file = await Deno.create("predictions.csv");
file.writeSync(new TextEncoder().encode(file_string));






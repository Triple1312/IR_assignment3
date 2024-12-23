import fs from 'node:fs';

import DataFrame from "./DataFrame.ts";
import RetrievalSystem2 from "./RetrievalSystem.ts";


let queries : DataFrame = await DataFrame.fromCSV("../data/docs_small/dev_small_queries.csv");
console.log("loaded queries");

let fileIndecesNames : string[] = []
let fileContents : string[] = [];

// load all documents
for await (const dirFileEntry of fs.readdirSync("../data/docs_small/full_docs_small")) {
  if (!dirFileEntry.endsWith(".txt")) {
    continue;
  }
  fileIndecesNames.push(dirFileEntry.split('_')[1].split('.')[0].toString());
  fileContents.push(fs.readFileSync("../data/docs_small/full_docs_small/" + dirFileEntry).toString());
}

console.log("loaded files");
let retrievalSystem : RetrievalSystem = new RetrievalSystem();

await retrievalSystem.fit(fileContents);

let finalFileString : string = "Query_number,doc_number";

let queryIndex : number = 0;

for (let i = 0; i < queries.data.length; i++) {
  let x: number[][] = await retrievalSystem.query(queries.data[i][1], 10);
  for (let j = 0; j < x.length; j++) {
    finalFileString += `\n${queries.data[i][0]},${fileIndecesNames[x[j][0]]}`;
  }
  queryIndex++;
  if (queryIndex % 100 == 0) {
    console.log(`Processed ${queryIndex} queries`);
  }
}

fs.writeFileSync("predictions.csv", finalFileString);



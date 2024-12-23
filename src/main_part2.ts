import DataFrame from "./DataFrame.js";
import fs from "node:fs";
import KMeansRetrievalSystem from "./KMeansRetrievalSystem.js";


let queries: DataFrame = await DataFrame.fromCSV("queries.csv", '\t');
console.log("loaded queries");

let fileIndecesNames: string[] = []
let fileContents: string[] = [];


let documentIndex = 0;

for await (const dirFileEntry of fs.readdirSync("../data/docs_large/full_docs")) {
  documentIndex++;
  if (!dirFileEntry.endsWith(".txt")) {
    continue;
  }
  fileIndecesNames.push(dirFileEntry.split('_')[1].split('.')[0].toString());
  fileContents.push(fs.readFileSync("../data/docs_large/full_docs/" + dirFileEntry).toString());
  if (documentIndex % 10000 == 0) {
    console.log(`Processed ${documentIndex} documents`);
    // break; // todo
  }
}

console.log("loaded files");

let kRetrievalSystem: KMeansRetrievalSystem = new KMeansRetrievalSystem();


await  kRetrievalSystem.fit(fileContents);

let finalFileString : string = "Query_number,doc_number";

let queryIndex : number = 0;

for (let i = 0; i < 1000; i++) {
  let x: number[][] = await kRetrievalSystem.query(queries.data[i][1], 10, 10);
  for (let j = 0; j < x.length; j++) {
    finalFileString += `\n${queries.data[i][0]},${fileIndecesNames[x[j][0]]}`;
  }
  queryIndex++;
  if (queryIndex % 100 == 0) {
    console.log(`Processed ${queryIndex} queries`);
  }
}

fs.writeFileSync("predictions.csv", finalFileString);
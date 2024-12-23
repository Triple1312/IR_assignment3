import fs from 'node:fs';

export default class DataFrame {
  _columnNames: string[] = [];
  _data: string[][] = [];


  private constructor() {}

  static async fromCSV(path: string, seperator = ',') : Promise<DataFrame> {
    const text = fs.readFileSync(path).toString();
    const text_lines = text.split("\n");
    let df : DataFrame = new DataFrame();
    df._columnNames = text_lines[0].split(seperator);
    for (let i = 1; i < text_lines.length; i++) {
      if (text_lines[i].length == 0) {
        continue;
      }
      df._data.push(text_lines[i].split(seperator));
    }
    return df;
  }

  get columnNames(): string[] {return this._columnNames};

  get data(): string[][] {return this._data};

  getRow(row: number): string[] {
    return this._data[row];
  }

}


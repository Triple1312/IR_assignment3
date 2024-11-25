


export default class DataFrame {
  _columnNames: string[] = [];
  _data: string[][] = [];


  private constructor() {}

  static async fromCSV(path: string) : Promise<DataFrame> {
    const text = await Deno.readTextFile(path);
    const text_lines = text.split("\n");
    let df : DataFrame = new DataFrame();
    df._columnNames = text_lines[0].split(",");
    for (let i = 1; i < text_lines.length; i++) {
      df._data.push(text_lines[i].split(","));
    }
    return df;
  }

  get columnNames(): string[] {return this._columnNames};

  get data(): string[][] {return this._data};

  getRow(row: number): string[] {
    return this._data[row];
  }

}






































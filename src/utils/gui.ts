// src/utils/gui.ts
export class GUIController {
  private inputEl!: HTMLInputElement;
  private valSpan!: HTMLSpanElement;
  private _name: string;
  private _onChange?: (v: number) => void;

  constructor(
    private parent: HTMLElement,
    private target: any,
    private prop: string,
    private min = 0,
    private max = 100,
    private step = 1
  ) {
    this._name = prop;
    this.createDOM();
  }

  private createDOM() {
    const row = document.createElement("div");
    row.style.cssText = "margin-bottom: 8px;";

    const labelRow = document.createElement("div");
    labelRow.style.cssText = "display: flex; justify-content: space-between; margin-bottom: 3px;";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = this._name;
    titleSpan.className = "gui-title";

    this.valSpan = document.createElement("span");
    this.valSpan.style.color = "#38bdf8";
    this.valSpan.textContent = Number(this.target[this.prop]).toFixed(2);

    labelRow.appendChild(titleSpan);
    labelRow.appendChild(this.valSpan);

    this.inputEl = document.createElement("input");
    this.inputEl.type = "range";
    this.inputEl.min = this.min.toString();
    this.inputEl.max = this.max.toString();
    this.inputEl.step = this.step.toString();
    this.inputEl.value = this.target[this.prop].toString();
    this.inputEl.style.cssText = "width: 100%; cursor: pointer; accent-color: #0284c7;";

    this.inputEl.oninput = (e) => {
      const val = parseFloat((e.target as HTMLInputElement).value);
      this.target[this.prop] = val;
      this.valSpan.textContent = val.toFixed(2);
      if (this._onChange) this._onChange(val);
    };

    row.appendChild(labelRow);
    row.appendChild(this.inputEl);
    this.parent.appendChild(row);
  }

  name(newName: string) {
    this._name = newName;
    const title = this.parent.querySelector(".gui-title");
    if (title) title.textContent = newName;
    return this;
  }

  onChange(fn: (v: number) => void) {
    this._onChange = fn;
    return this;
  }

  updateDisplay() {
    const val = Number(this.target[this.prop]);
    this.inputEl.value = val.toString();
    this.valSpan.textContent = val.toFixed(2);
  }
}

export class SimpleGUI {
  addFolder(arg0: string) {
      throw new Error("Method not implemented.");
  }
  private controllers: GUIController[] = [];

  constructor(private container: HTMLElement) {}

  add(target: any, prop: string, min = 0, max = 100, step = 1): GUIController {
    this.show();
    const ctrl = new GUIController(this.container, target, prop, min, max, step);
    this.controllers.push(ctrl);
    return ctrl;
  }

  addButton(name: string, onClick: () => void) {
    this.show();
    const btn = document.createElement("button");
    btn.textContent = name;
    btn.style.cssText = `
      width: 100%; padding: 6px; margin: 4px 0; background: #0284c7;
      border: none; border-radius: 4px; color: #fff; cursor: pointer;
      font-weight: 500; font-size: 12px; transition: 0.15s;
    `;
    btn.onmouseover = () => (btn.style.background = "#0369a1");
    btn.onmouseout = () => (btn.style.background = "#0284c7");
    btn.onclick = onClick;
    this.container.appendChild(btn);
  }

  addTextInfo(info: string) {
    this.show();
    const p = document.createElement("div");
    p.innerHTML = info;
    p.style.cssText = "font-size: 11px; color: #71717a; margin-top: 8px; line-height: 1.4; border-top: 1px solid #3f3f46; padding-top: 6px;";
    this.container.appendChild(p);
  }

  updateDisplay() {
    this.controllers.forEach((c) => c.updateDisplay());
  }

  clear() {
    this.container.innerHTML = "";
    this.controllers = [];
    this.hide();
  }

  private show() {
    this.container.style.display = "block";
  }

  private hide() {
    this.container.style.display = "none";
  }
}
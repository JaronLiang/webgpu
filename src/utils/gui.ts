// src/utils/gui.ts

export type ControllerType = "number" | "boolean" | "select" | "function";

export class GUIController {
  private domElement!: HTMLElement;
  private inputEl?: HTMLInputElement | HTMLSelectElement;
  private valSpan?: HTMLSpanElement;
  private titleSpan!: HTMLSpanElement;
  private _name: string;
  private _onChange?: (v: any) => void;
  private _type: ControllerType = "number";
  private _isListening = false;

  constructor(
    private parent: HTMLElement,
    private target: any,
    private prop: string,
    private arg1?: any,
    private arg2?: any,
    private arg3?: any
  ) {
    this._name = prop;
    this.determineType();
    this.createDOM();
  }

  private determineType() {
    const val = this.target[this.prop];
    if (typeof val === "function") {
      this._type = "function";
    } else if (Array.isArray(this.arg1) || (typeof this.arg1 === "object" && this.arg1 !== null && typeof this.arg2 === "undefined")) {
      this._type = "select";
    } else if (typeof val === "boolean") {
      this._type = "boolean";
    } else {
      this._type = "number";
    }
  }

  private createDOM() {
    const row = document.createElement("div");
    row.style.cssText = "margin-bottom: 8px; font-size: 12px; color: #e2e8f0; font-family: sans-serif;";
    this.domElement = row;

    if (this._type === "function") {
      // 1. 函数渲染为按钮
      const btn = document.createElement("button");
      btn.textContent = this._name;
      btn.style.cssText = `
        width: 100%; padding: 6px 10px; background: #0284c7; border: none;
        border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;
        transition: 0.15s; font-weight: 500;
      `;
      btn.onmouseover = () => (btn.style.background = "#0369a1");
      btn.onmouseout = () => (btn.style.background = "#0284c7");
      btn.onclick = () => {
        this.target[this.prop]();
        if (this._onChange) this._onChange(this.target[this.prop]);
      };
      row.appendChild(btn);
      this.parent.appendChild(row);
      return;
    }

    // 标题行
    const labelRow = document.createElement("div");
    labelRow.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;";

    this.titleSpan = document.createElement("span");
    this.titleSpan.textContent = this._name;
    this.titleSpan.style.cssText = "user-select: none; opacity: 0.9;";
    labelRow.appendChild(this.titleSpan);

    if (this._type === "number") {
      // 2. 数值滑块
      const min = typeof this.arg1 === "number" ? this.arg1 : 0;
      const max = typeof this.arg2 === "number" ? this.arg2 : 100;
      const step = typeof this.arg3 === "number" ? this.arg3 : (max - min) / 100;

      this.valSpan = document.createElement("span");
      this.valSpan.style.cssText = "color: #38bdf8; font-family: monospace; font-size: 11px;";
      const curNum = Number(this.target[this.prop]);
      this.valSpan.textContent = isNaN(curNum) ? "0.00" : curNum.toFixed(2);
      labelRow.appendChild(this.valSpan);
      row.appendChild(labelRow);

      const input = document.createElement("input");
      input.type = "range";
      input.min = min.toString();
      input.max = max.toString();
      input.step = step.toString();
      input.value = (this.target[this.prop] ?? min).toString();
      input.style.cssText = "width: 100%; cursor: pointer; accent-color: #0284c7; margin: 0;";

      input.oninput = (e) => {
        const val = parseFloat((e.target as HTMLInputElement).value);
        this.target[this.prop] = val;
        if (this.valSpan) this.valSpan.textContent = val.toFixed(2);
        if (this._onChange) this._onChange(val);
      };

      this.inputEl = input;
      row.appendChild(input);
    } else if (this._type === "boolean") {
      // 3. 布尔开关
      const checkWrapper = document.createElement("label");
      checkWrapper.style.cssText = "display: flex; align-items: center; cursor: pointer;";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(this.target[this.prop]);
      input.style.cssText = "cursor: pointer; accent-color: #0284c7; width: 15px; height: 15px; margin: 0;";

      input.onchange = (e) => {
        const checked = (e.target as HTMLInputElement).checked;
        this.target[this.prop] = checked;
        if (this._onChange) this._onChange(checked);
      };

      this.inputEl = input;
      labelRow.appendChild(input);
      row.appendChild(labelRow);
    } else if (this._type === "select") {
      // 4. 下拉选项列表 (Select)
      row.appendChild(labelRow);

      const select = document.createElement("select");
      select.style.cssText = `
        width: 100%; padding: 4px 6px; background: #1e293b; color: #38bdf8;
        border: 1px solid #334155; border-radius: 4px; outline: none; cursor: pointer;
      `;

      const options: string[] = Array.isArray(this.arg1)
        ? this.arg1
        : Object.keys(this.arg1);

      for (const opt of options) {
        const optEl = document.createElement("option");
        optEl.value = opt;
        optEl.textContent = opt;
        if (String(this.target[this.prop]) === String(opt)) {
          optEl.selected = true;
        }
        select.appendChild(optEl);
      }

      select.onchange = (e) => {
        const val = (e.target as HTMLSelectElement).value;
        this.target[this.prop] = val;
        if (this._onChange) this._onChange(val);
      };

      this.inputEl = select;
      row.appendChild(select);
    }

    this.parent.appendChild(row);
  }

  name(newName: string) {
    this._name = newName;
    if (this.titleSpan) {
      this.titleSpan.textContent = newName;
    } else {
      const btn = this.domElement.querySelector("button");
      if (btn) btn.textContent = newName;
    }
    return this;
  }

  onChange<T = any>(fn: (v: T) => void) {
    this._onChange = fn;
    return this;
  }

  listen() {
    this._isListening = true;
    return this;
  }

  updateDisplay() {
    if (!this.inputEl) return;
    const val = this.target[this.prop];

    if (this._type === "number") {
      const num = Number(val);
      this.inputEl.value = num.toString();
      if (this.valSpan) this.valSpan.textContent = isNaN(num) ? "0.00" : num.toFixed(2);
    } else if (this._type === "boolean") {
      (this.inputEl as HTMLInputElement).checked = Boolean(val);
    } else if (this._type === "select") {
      (this.inputEl as HTMLSelectElement).value = String(val);
    }
  }

  get isListening() {
    return this._isListening;
  }
}

export class GUIFolder {
  private folderContent: HTMLElement;
  private controllers: GUIController[] = [];

  constructor(private parent: HTMLElement, public title: string) {
    const container = document.createElement("div");
    container.style.cssText = "margin-bottom: 6px; border: 1px solid #334155; border-radius: 6px; overflow: hidden;";

    const header = document.createElement("div");
    header.style.cssText = `
      background: #1e293b; padding: 6px 8px; font-size: 11px; font-weight: bold;
      color: #94a3b8; cursor: pointer; display: flex; justify-content: space-between; user-select: none;
    `;
    header.innerHTML = `<span>📁 ${title}</span><span class="gui-arrow">▼</span>`;

    this.folderContent = document.createElement("div");
    this.folderContent.style.cssText = "padding: 8px; background: rgba(15, 23, 42, 0.4);";

    let isOpen = true;
    header.onclick = () => {
      isOpen = !isOpen;
      this.folderContent.style.display = isOpen ? "block" : "none";
      const arrow = header.querySelector(".gui-arrow");
      if (arrow) arrow.textContent = isOpen ? "▼" : "◀";
    };

    container.appendChild(header);
    container.appendChild(this.folderContent);
    this.parent.appendChild(container);
  }

  add(target: any, prop: string, arg1?: any, arg2?: any, arg3?: any): GUIController {
    const ctrl = new GUIController(this.folderContent, target, prop, arg1, arg2, arg3);
    this.controllers.push(ctrl);
    return ctrl;
  }

  updateDisplay() {
    this.controllers.forEach((c) => c.updateDisplay());
  }
}

export class SimpleGUI {
  private controllers: GUIController[] = [];
  private folders: GUIFolder[] = [];

  constructor(private container: HTMLElement) {}

  addFolder(title: string): GUIFolder {
    this.show();
    const folder = new GUIFolder(this.container, title);
    this.folders.push(folder);
    return folder;
  }

  add(target: any, prop: string, arg1?: any, arg2?: any, arg3?: any): GUIController {
    this.show();
    const ctrl = new GUIController(this.container, target, prop, arg1, arg2, arg3);
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
    p.style.cssText = "font-size: 11px; color: #94a3b8; margin-top: 8px; line-height: 1.5; border-top: 1px solid #334155; padding-top: 6px;";
    this.container.appendChild(p);
  }

  updateDisplay() {
    this.controllers.forEach((c) => c.updateDisplay());
    this.folders.forEach((f) => f.updateDisplay());
  }

  clear() {
    this.container.innerHTML = "";
    this.controllers = [];
    this.folders = [];
    this.hide();
  }

  private show() {
    this.container.style.display = "block";
  }

  private hide() {
    this.container.style.display = "none";
  }
}
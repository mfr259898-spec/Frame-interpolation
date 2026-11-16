package com.example.loadstore;

import android.app.Activity;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.*;

public class MainActivity extends Activity {
    static { System.loadLibrary("loadstore"); }

    native String nativeInfo();
    native String nativeLastError();
    native double[] nativeRun(int load, int store, int fmt, int samples, int res, int iters, double seconds);

    static final String[] LOAD = {"LOAD", "CLEAR", "DONT_CARE", "NONE"};
    static final String[] STORE = {"STORE", "DONT_CARE", "NONE"};
    static final String[] FMT = {"RGBA8", "RGBA16F", "RGBA32F"};
    static final String[] SAMP = {"1x", "2x", "4x"};
    static final String[] RES = {"1280x720", "1920x1080", "2560x1440", "3840x2160"};
    static final int[] SWEEP = {0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400};

    Spinner sLoad, sStore, sFmt, sSamp, sRes;
    EditText eIters;
    TextView out;
    Button[] buttons;
    volatile boolean busy = false;

    Spinner spin(LinearLayout parent, String label, String[] items, int sel) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        TextView t = new TextView(this);
        t.setText(label);
        t.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        Spinner s = new Spinner(this);
        s.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items));
        s.setSelection(sel);
        s.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2));
        row.addView(t);
        row.addView(s);
        parent.addView(row);
        return s;
    }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ScrollView sv = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int p = (int) (12 * getResources().getDisplayMetrics().density);
        root.setPadding(p, p * 2, p, p);
        sv.addView(root);

        sLoad = spin(root, "loadOp", LOAD, 0);
        sStore = spin(root, "storeOp", STORE, 0);
        sFmt = spin(root, "Format", FMT, 0);
        sSamp = spin(root, "MSAA", SAMP, 0);
        sRes = spin(root, "Resolution", RES, 1);

        LinearLayout row = new LinearLayout(this);
        TextView t = new TextView(this);
        t.setText("ALU iterations");
        t.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        eIters = new EditText(this);
        eIters.setText("1000");
        eIters.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        eIters.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2));
        row.addView(t);
        row.addView(eIters);
        root.addView(row);

        Button bRun = new Button(this); bRun.setText("Run once");
        Button bSweep = new Button(this); bSweep.setText("Sweep ALU (chosen vs DONT_CARE/DONT_CARE)");
        Button bMat = new Button(this); bMat.setText("All loadOp x storeOp combos");
        buttons = new Button[]{bRun, bSweep, bMat};
        root.addView(bRun); root.addView(bSweep); root.addView(bMat);

        out = new TextView(this);
        out.setTypeface(Typeface.MONOSPACE);
        out.setTextSize(11);
        out.setTextIsSelectable(true);
        root.addView(out);
        setContentView(sv);

        bRun.setOnClickListener(v -> start(0));
        bSweep.setOnClickListener(v -> start(1));
        bMat.setOnClickListener(v -> start(2));

        out.setText("Initializing Vulkan...\n");
        new Thread(() -> { String info = nativeInfo(); ui(info + "\n\n"); }).start();
    }

    void ui(String s) { runOnUiThread(() -> out.append(s)); }

    // returns ms/pass, or -1 on error (error text appended)
    double run(int l, int s, int f, int m, int r, int it, StringBuilder note) {
        double[] a = nativeRun(l, s, f, m, r, it, 1.0);
        if (a == null) { note.append("ERR: ").append(nativeLastError()); return -1; }
        note.append(String.format("(%d passes/submit, %.0f MB color)", (int) a[1], a[2]));
        return a[0];
    }

    void start(int mode) {
        if (busy) return;
        final int l = sLoad.getSelectedItemPosition(), s = sStore.getSelectedItemPosition(),
                f = sFmt.getSelectedItemPosition(), m = sSamp.getSelectedItemPosition(),
                r = sRes.getSelectedItemPosition();
        int itTmp = 1000;
        try { itTmp = Integer.parseInt(eIters.getText().toString().trim()); } catch (Exception e) {}
        final int it = Math.max(0, itTmp);
        busy = true;
        for (Button x : buttons) x.setEnabled(false);
        new Thread(() -> {
            String cfg = String.format("%s %s %s %s", FMT[f], SAMP[m], RES[r], "");
            try {
                if (mode == 0) {
                    ui(String.format("== %s/%s  %s  iters=%d\n", LOAD[l], STORE[s], cfg.trim(), it));
                    StringBuilder n = new StringBuilder();
                    double ms = run(l, s, f, m, r, it, n);
                    ui(ms < 0 ? n + "\n\n" : String.format("%.4f ms/pass  %s\n\n", ms, n));
                } else if (mode == 1) {
                    ui(String.format("== ALU sweep  %s  chosen=%s/%s  baseline=DONT_CARE/DONT_CARE\n", cfg.trim(), LOAD[l], STORE[s]));
                    ui(String.format("%6s %10s %10s %10s\n", "iters", "chosen", "baseline", "delta"));
                    for (int a : SWEEP) {
                        StringBuilder n1 = new StringBuilder(), n2 = new StringBuilder();
                        double c = run(l, s, f, m, r, a, n1);
                        double base = run(2, 1, f, m, r, a, n2);
                        if (c < 0 || base < 0) { ui(a + ": " + (c < 0 ? n1 : n2) + "\n"); continue; }
                        ui(String.format("%6d %10.4f %10.4f %+10.4f\n", a, c, base, c - base));
                    }
                    ui("(ms per render pass)\n\n");
                } else {
                    ui(String.format("== load/store matrix  %s  iters=%d\n", cfg.trim(), it));
                    for (int li = 0; li < LOAD.length; li++)
                        for (int si = 0; si < STORE.length; si++) {
                            StringBuilder n = new StringBuilder();
                            double ms = run(li, si, f, m, r, it, n);
                            ui(ms < 0 ? String.format("%-9s %-9s %s\n", LOAD[li], STORE[si], n)
                                      : String.format("%-9s %-9s %9.4f ms\n", LOAD[li], STORE[si], ms));
                        }
                    ui("\n");
                }
            } finally {
                busy = false;
                runOnUiThread(() -> { for (Button x : buttons) x.setEnabled(true); });
            }
        }).start();
    }
}

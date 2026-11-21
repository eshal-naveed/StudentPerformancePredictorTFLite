package com.example.numberguessingtflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.ColorStateList;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.android.material.card.MaterialCardView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Student Performance Predictor
 * An on-device TensorFlow Lite model predicts the probability that a student
 * will pass, based on study hours, attendance and previous marks.
 * The app also shows a risk level and personalised advice.
 */
public class MainActivity extends AppCompatActivity {

    // ---------- UI elements ----------
    private EditText etHours, etAttendance, etMarks;
    private Button btnPredict, btnReset;
    private TextView tvVerdict, tvProb, tvRisk, tvAdvice;
    private ProgressBar progressBar;
    private MaterialCardView cardResult;

    // TFLite Interpreter: runs the model on the device
    private Interpreter interpreter;

    // Name of the model file stored in app/src/main/assets/
    private static final String MODEL_FILE = "student_model.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Apply padding for the status bar and navigation bar
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });

        // Connect XML views to Java variables
        etHours = findViewById(R.id.etHours);
        etAttendance = findViewById(R.id.etAttendance);
        etMarks = findViewById(R.id.etMarks);
        btnPredict = findViewById(R.id.btnPredict);
        btnReset = findViewById(R.id.btnReset);
        tvVerdict = findViewById(R.id.tvVerdict);
        tvProb = findViewById(R.id.tvProb);
        tvRisk = findViewById(R.id.tvRisk);
        tvAdvice = findViewById(R.id.tvAdvice);
        progressBar = findViewById(R.id.progressBar);
        cardResult = findViewById(R.id.cardResult);

        // STEP 1: Load the model file and create the Interpreter
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load model: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        btnPredict.setOnClickListener(v -> predict());
        btnReset.setOnClickListener(v -> resetAll());
    }

    /** Reads the input, validates it, runs the model and shows the result. */
    private void predict() {
        float hours, attendance, marks;
        try {
            hours = Float.parseFloat(etHours.getText().toString().trim());
            attendance = Float.parseFloat(etAttendance.getText().toString().trim());
            marks = Float.parseFloat(etMarks.getText().toString().trim());
        } catch (NumberFormatException e) {
            Toast.makeText(this, "Please enter valid numbers in all fields",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        // Range validation
        if (hours < 0 || hours > 10 || attendance < 0 || attendance > 100
                || marks < 0 || marks > 100) {
            Toast.makeText(this,
                    "Hours must be 0-10, attendance and marks must be 0-100",
                    Toast.LENGTH_LONG).show();
            return;
        }

        // STEP 2: Prepare the input. Shape [1][3], type float32.
        // The values were scaled to 0-1 during training, so we scale them here too.
        float[][] input = new float[1][3];
        input[0][0] = hours / 10f;
        input[0][1] = attendance / 100f;
        input[0][2] = marks / 100f;

        // Output buffer: shape [1][1] (probability of passing, between 0 and 1)
        float[][] output = new float[1][1];

        // STEP 3: Run inference
        interpreter.run(input, output);

        // STEP 4: Interpret the output
        float prob = output[0][0];
        boolean pass = prob >= 0.5f;
        int percent = Math.round(prob * 100);

        // STEP 5: Update the UI
        int green = 0xFF2E7D32, orange = 0xFFEF6C00, red = 0xFFC62828;
        int color;
        String risk;
        if (prob >= 0.75f) {
            risk = "Risk Level: SAFE";
            color = green;
        } else if (prob >= 0.5f) {
            risk = "Risk Level: MODERATE";
            color = orange;
        } else {
            risk = "Risk Level: HIGH RISK";
            color = red;
        }

        tvVerdict.setText(pass ? "PASS" : "FAIL");
        tvVerdict.setTextColor(color);
        tvProb.setText("Pass probability: " + String.format("%.1f", prob * 100) + "%");
        progressBar.setProgress(percent);
        progressBar.setProgressTintList(ColorStateList.valueOf(color));
        tvRisk.setText(risk);
        tvRisk.setTextColor(color);
        tvAdvice.setText(buildAdvice(hours, attendance, marks, prob));
        cardResult.setVisibility(View.VISIBLE);
    }

    /** Rule-based advice: finds the weak factors and gives tips. */
    private String buildAdvice(float hours, float attendance, float marks, float prob) {
        StringBuilder sb = new StringBuilder();
        if (attendance < 75) {
            sb.append("• Your attendance is below 75%. Attend more classes.\n");
        }
        if (hours < 4) {
            sb.append("• Increase your daily study time (at least 4 hours).\n");
        }
        if (marks < 50) {
            sb.append("• Your previous marks are low. Revise the weak topics.\n");
        }
        if (sb.length() == 0) {
            sb.append(prob >= 0.75f
                    ? "• Great job! Keep up the same routine."
                    : "• A little more effort will move you into the safe zone.");
        }
        return sb.toString().trim();
    }

    /** Clears all input fields and hides the result card. */
    private void resetAll() {
        etHours.setText("");
        etAttendance.setText("");
        etMarks.setText("");
        cardResult.setVisibility(View.GONE);
        etHours.requestFocus();
    }

    /** Maps the .tflite file from the assets folder into memory. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fd = getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = inputStream.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getLength());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close(); // free memory
        }
    }
}
# Import Experiment Data

This tutorial shows how to import an experiment `.mat` file into IMRFit.

## Standard MAT Import

[Watch demo](https://github.com/sicong0613/IMR_bubble_GUI/issues/1#issuecomment-4626402986)

1. Open `File -> Import Wizard...`.
2. Click `Import` or drag a `.mat` file into the drop area.
3. Map the experimental curve variables. The GUI recognizes common radius names such as `R_exp`, `R`, `radius`, `radius_exp`, and `R1_exp`.
4. Check units for time and radius.
5. Click `Import data`.

## Raw Pixel-Based Data

[Watch demo](https://github.com/sicong0613/IMR_bubble_GUI/issues/1#issuecomment-4626407825)

1. Open `File -> Import Wizard...`.
2. Click `Import` or drag a `.mat` file into the drop area.
3. Map `R_exp` to the radius variable.
4. Set the radius unit to `pixel`.
5. Enter the `um/pixel` calibration.
6. If the file has no time-axis variable, leave `t_exp` as `(none)` and enter the camera `fps`.
7. Click "Learn names" to save the current settings to profile.
8. Click `Import data`.

# After clicking "Learn names", you may now import the MAT file with Standard MAT import steps, which is the base for all following batch process.


## Notes

- `Learn names` saves the selected variable names for future recognition.
- `Remove negative R` and `Remove isolated spikes` can clean common tracking artifacts during import.
- If Curve View multiple selection is enabled, imported curves are added to Curve View instead of replacing the current preview.

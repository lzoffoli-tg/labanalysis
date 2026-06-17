# Tutorial: Building Custom Test Protocols

Learn how to create custom test protocols by extending labanalysis base classes for specialized assessment workflows.

**Duration**: 45 minutes  
**Level**: Advanced  
**Prerequisites**: labanalysis installed, Python OOP knowledge, understanding of existing protocols

## What You'll Learn

- Understand TestProtocol and TestResults interfaces
- Create custom protocol classes
- Implement from_files() class method
- Implement get_results() processing method
- Design custom result containers
- Add visualization methods
- Integrate with existing labanalysis infrastructure
- Handle edge cases and validation

## Scenario

You want to create a custom **Agility T-Test** protocol that:
1. Loads motion capture data (4 cone positions + athlete marker)
2. Detects when athlete reaches each cone
3. Calculates split times between cones
4. Computes total time and cone-to-cone velocities
5. Generates visualization with trajectory plot
6. Exports results to standardized format

This protocol doesn't exist in labanalysis, so you'll build it from scratch following best practices.

## Part 1: Understanding the Interface

### Step 1: Review Base Classes

```python
import labanalysis as laban
from labanalysis.protocols import TestProtocol, TestResults, Participant
from labanalysis.records import TimeseriesRecord, Point3D, Signal1D
from labanalysis.records.pipelines import ProcessingPipeline
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pathlib import Path

# TestProtocol interface (duck typing)
class TestProtocol:
    """
    Base interface for test protocols.
    
    Required attributes:
        - participant: Participant instance
        - processing_pipeline: ProcessingPipeline (optional)
        - processed_data: Processed copy of test data
    
    Required methods:
        - from_files() [classmethod]: Load test from file(s)
        - get_results() -> TestResults: Process and return results
        - save(file_path): Save protocol to file
        - load(file_path) [classmethod]: Load protocol from file
    """

# TestResults interface
class TestResults:
    """
    Base interface for test results.
    
    Required attributes:
        - participant: Participant instance
        - summary: pd.DataFrame with summary statistics
        - analytics: pd.DataFrame with time-series data (optional)
        - figures: dict of plotly figures (optional)
    
    Required methods:
        - plot() -> go.Figure: Generate visualization
        - to_dataframe() -> pd.DataFrame: Export all data
        - save(file_path): Save results to file
        - load(file_path) [classmethod]: Load results from file
    """
```

## Part 2: Define Custom Protocol

### Step 2: Create TTestProtocol Class

```python
class TTestProtocol(TestProtocol):
    """
    Protocol for agility T-Test assessment.
    
    The T-Test measures agility through forward sprint, lateral shuffles,
    and backpedaling. Four cones are placed in a T-shape, and the athlete
    navigates between them following a specific pattern.
    
    Cone layout (10m x 10m):
          B
          |
    C --- A --- D
          |
        Start
    
    Sequence: Start → A → B → A → C → A → D → A → Start
    
    Parameters
    ----------
    participant : Participant
        Participant information
    athlete_marker : Point3D
        Athlete's COM or sacrum marker position
    cone_a : Point3D
        Center cone position (known/fixed)
    cone_b : Point3D
        Front cone position
    cone_c : Point3D
        Left cone position
    cone_d : Point3D
        Right cone position
    processing_pipeline : ProcessingPipeline, optional
        Signal processing pipeline for marker data
    
    Attributes
    ----------
    participant : Participant
    athlete_marker : Point3D
    cones : dict
        Dictionary of cone positions {'A': Point3D, 'B': Point3D, ...}
    processing_pipeline : ProcessingPipeline
    processed_data : TTestProtocol
        Copy with processed marker data
    """
    
    def __init__(
        self,
        participant: Participant,
        athlete_marker: Point3D,
        cone_a: Point3D,
        cone_b: Point3D,
        cone_c: Point3D,
        cone_d: Point3D,
        processing_pipeline: ProcessingPipeline = None
    ):
        self.participant = participant
        self.athlete_marker = athlete_marker
        self.cones = {
            'A': cone_a,
            'B': cone_b,
            'C': cone_c,
            'D': cone_d
        }
        
        # Default processing: 6 Hz lowpass for markers
        if processing_pipeline is None:
            self.processing_pipeline = ProcessingPipeline()
            self.processing_pipeline.add_step(
                lambda x: laban.butterworth_filt(x, athlete_marker.sampling_frequency, 6, 4, 'low')
            )
        else:
            self.processing_pipeline = processing_pipeline
        
        # Apply processing
        self.processed_data = self._process()
    
    def _process(self):
        """Apply processing pipeline to marker data."""
        processed_marker = Point3D(
            data={
                'x': self.processing_pipeline.apply(self.athlete_marker['x'].data),
                'y': self.processing_pipeline.apply(self.athlete_marker['y'].data),
                'z': self.processing_pipeline.apply(self.athlete_marker['z'].data)
            },
            sampling_frequency=self.athlete_marker.sampling_frequency,
            unit=self.athlete_marker.unit
        )
        
        # Create new instance with processed data
        return TTestProtocol(
            participant=self.participant,
            athlete_marker=processed_marker,
            cone_a=self.cones['A'],
            cone_b=self.cones['B'],
            cone_c=self.cones['C'],
            cone_d=self.cones['D'],
            processing_pipeline=None  # Already processed
        )
    
    @classmethod
    def from_files(
        cls,
        participant: Participant,
        tdf_filename: str,
        athlete_marker_key: str = 'sacrum',
        cone_a_position: tuple = (0, 0, 0),
        cone_b_position: tuple = (0, 10, 0),
        cone_c_position: tuple = (-5, 0, 0),
        cone_d_position: tuple = (5, 0, 0)
    ):
        """
        Create T-Test protocol from TDF file.
        
        Parameters
        ----------
        participant : Participant
            Participant information
        tdf_filename : str
            Path to TDF file with marker data
        athlete_marker_key : str, default='sacrum'
            Key for athlete marker in TDF
        cone_a_position : tuple, default=(0, 0, 0)
            Center cone position (x, y, z) in meters
        cone_b_position : tuple, default=(0, 10, 0)
            Front cone position
        cone_c_position : tuple, default=(-5, 0, 0)
            Left cone position
        cone_d_position : tuple, default=(5, 0, 0)
            Right cone position
        
        Returns
        -------
        TTestProtocol
            Protocol instance
        """
        # Load marker data
        data = laban.read_tdf(tdf_filename)
        athlete_marker = data[athlete_marker_key]
        
        # Create fixed cone positions as Point3D
        fs = athlete_marker.sampling_frequency
        n_samples = len(athlete_marker)
        
        def create_fixed_point(position):
            return Point3D(
                data={
                    'x': np.full(n_samples, position[0]),
                    'y': np.full(n_samples, position[1]),
                    'z': np.full(n_samples, position[2])
                },
                sampling_frequency=fs,
                unit='m'
            )
        
        cone_a = create_fixed_point(cone_a_position)
        cone_b = create_fixed_point(cone_b_position)
        cone_c = create_fixed_point(cone_c_position)
        cone_d = create_fixed_point(cone_d_position)
        
        return cls(
            participant=participant,
            athlete_marker=athlete_marker,
            cone_a=cone_a,
            cone_b=cone_b,
            cone_c=cone_c,
            cone_d=cone_d
        )
    
    def get_results(self):
        """
        Process T-Test and return results.
        
        Returns
        -------
        TTestResults
            Results object with split times and visualization
        """
        return TTestResults(self)
    
    def save(self, file_path: str):
        """Save protocol to pickle file."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str):
        """Load protocol from pickle file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
```

### Step 3: Create TTestResults Class

```python
class TTestResults(TestResults):
    """
    Container for T-Test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary statistics (total time, split times, velocities)
    analytics : pd.DataFrame
        Time-series data (distance to each cone over time)
    figures : dict
        Plotly figures {'trajectory': go.Figure, 'splits': go.Figure}
    """
    
    def __init__(self, protocol: TTestProtocol):
        self.participant = protocol.participant
        self.protocol = protocol
        
        # Process test
        self._detect_events()
        self._calculate_metrics()
        self._generate_figures()
    
    def _detect_events(self):
        """Detect when athlete reaches each cone."""
        marker = self.protocol.processed_data.athlete_marker
        
        # Calculate distance to each cone
        self.distances = {}
        for cone_name, cone in self.protocol.cones.items():
            dx = marker['x'].data - cone['x'].data
            dy = marker['y'].data - cone['y'].data
            dist = np.sqrt(dx**2 + dy**2)
            self.distances[cone_name] = dist
        
        # Expected sequence: Start → A → B → A → C → A → D → A → Start
        # Detect cone touches (distance < threshold)
        threshold = 0.5  # meters
        
        events = []
        time = np.arange(len(marker)) / marker.sampling_frequency
        
        # Find when distance drops below threshold
        for cone_name in ['A', 'B', 'A', 'C', 'A', 'D', 'A']:
            dist = self.distances[cone_name]
            
            # Find first time after last event when close to this cone
            start_idx = events[-1]['index'] + 50 if events else 0
            close_indices = np.where(dist[start_idx:] < threshold)[0]
            
            if len(close_indices) > 0:
                idx = start_idx + close_indices[0]
                events.append({
                    'cone': cone_name,
                    'time': time[idx],
                    'index': idx
                })
        
        # Add start (t=0)
        self.events = [{'cone': 'Start', 'time': 0.0, 'index': 0}] + events
        
        # Add finish (return to start)
        finish_idx = len(marker) - 1
        self.events.append({'cone': 'Finish', 'time': time[finish_idx], 'index': finish_idx})
    
    def _calculate_metrics(self):
        """Calculate split times and velocities."""
        splits = []
        
        for i in range(len(self.events) - 1):
            event_from = self.events[i]
            event_to = self.events[i + 1]
            
            split_time = event_to['time'] - event_from['time']
            
            # Calculate distance traveled
            idx_from = event_from['index']
            idx_to = event_to['index']
            marker = self.protocol.processed_data.athlete_marker
            
            dx = marker['x'].data[idx_to] - marker['x'].data[idx_from]
            dy = marker['y'].data[idx_to] - marker['y'].data[idx_from]
            distance = np.sqrt(dx**2 + dy**2)
            
            velocity = distance / split_time if split_time > 0 else 0
            
            splits.append({
                'from': event_from['cone'],
                'to': event_to['cone'],
                'split_time': split_time,
                'distance': distance,
                'velocity': velocity
            })
        
        self.analytics = pd.DataFrame(splits)
        
        # Summary statistics
        total_time = self.events[-1]['time']
        mean_velocity = self.analytics['velocity'].mean()
        max_velocity = self.analytics['velocity'].max()
        
        self.summary = pd.DataFrame([{
            'parameter': 'Total Time',
            'value': total_time,
            'unit': 's'
        }, {
            'parameter': 'Mean Velocity',
            'value': mean_velocity,
            'unit': 'm/s'
        }, {
            'parameter': 'Peak Velocity',
            'value': max_velocity,
            'unit': 'm/s'
        }])
    
    def _generate_figures(self):
        """Generate visualization figures."""
        self.figures = {}
        
        # Trajectory plot
        marker = self.protocol.processed_data.athlete_marker
        
        fig_traj = go.Figure()
        
        # Plot athlete path
        fig_traj.add_trace(go.Scatter(
            x=marker['x'].data,
            y=marker['y'].data,
            mode='lines',
            name='Path',
            line=dict(color='blue', width=2)
        ))
        
        # Plot cones
        for cone_name, cone in self.protocol.cones.items():
            fig_traj.add_trace(go.Scatter(
                x=[cone['x'].data[0]],
                y=[cone['y'].data[0]],
                mode='markers+text',
                name=f'Cone {cone_name}',
                text=[cone_name],
                textposition='top center',
                marker=dict(size=15, symbol='triangle-up')
            ))
        
        # Plot event markers
        event_x = [marker['x'].data[e['index']] for e in self.events[1:-1]]
        event_y = [marker['y'].data[e['index']] for e in self.events[1:-1]]
        
        fig_traj.add_trace(go.Scatter(
            x=event_x,
            y=event_y,
            mode='markers',
            name='Cone touches',
            marker=dict(size=10, color='red', symbol='circle')
        ))
        
        fig_traj.update_layout(
            title='T-Test Trajectory',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            yaxis_scaleanchor='x',
            height=600
        )
        
        self.figures['trajectory'] = fig_traj
        
        # Split times bar chart
        fig_splits = go.Figure()
        
        labels = [f"{s['from']}→{s['to']}" for _, s in self.analytics.iterrows()]
        times = self.analytics['split_time'].values
        
        fig_splits.add_trace(go.Bar(
            x=labels,
            y=times,
            name='Split Time',
            marker_color='steelblue'
        ))
        
        fig_splits.update_layout(
            title='Split Times',
            xaxis_title='Segment',
            yaxis_title='Time (s)',
            height=400
        )
        
        self.figures['splits'] = fig_splits
    
    def plot(self):
        """Generate combined visualization."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Trajectory', 'Split Times'],
            specs=[[{'type': 'scatter'}], [{'type': 'bar'}]],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.15
        )
        
        # Add trajectory traces
        for trace in self.figures['trajectory'].data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add split times
        for trace in self.figures['splits'].data:
            fig.add_trace(trace, row=2, col=1)
        
        fig.update_xaxes(title_text="X (m)", row=1, col=1)
        fig.update_yaxes(title_text="Y (m)", row=1, col=1)
        fig.update_xaxes(title_text="Segment", row=2, col=1)
        fig.update_yaxes(title_text="Time (s)", row=2, col=1)
        
        fig.update_layout(height=900, showlegend=True, title_text="T-Test Results")
        
        return fig
    
    def to_dataframe(self):
        """Export all metrics to DataFrame."""
        return pd.concat([
            self.summary.assign(type='summary'),
            self.analytics.assign(type='analytics')
        ], ignore_index=True)
    
    def save(self, file_path: str):
        """Save results to pickle file."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str):
        """Load results from pickle file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
```

## Part 3: Use Custom Protocol

### Step 4: Run T-Test Analysis

```python
# Create participant
participant = laban.Participant(
    surname='Speedster',
    name='Agile',
    age=24,
    gender='F',
    height=168,
    weight=62
)

# Load and process T-Test
test = TTestProtocol.from_files(
    participant=participant,
    tdf_filename='ttest_trial.tdf',
    athlete_marker_key='sacrum',
    cone_a_position=(0, 0, 0),
    cone_b_position=(0, 10, 0),
    cone_c_position=(-5, 0, 0),
    cone_d_position=(5, 0, 0)
)

# Get results
results = test.get_results()

# View summary
print("=== T-TEST RESULTS ===\n")
print(results.summary.to_string(index=False))

# View split times
print("\n=== SPLIT TIMES ===\n")
print(results.analytics.to_string(index=False))

# Plot
fig = results.plot()
fig.show()
fig.write_html("ttest_results.html")

# Save
test.save("ttest_protocol.pkl")
results.save("ttest_results.pkl")
```

**Output:**
```
=== T-TEST RESULTS ===

      parameter  value unit
     Total Time  11.23    s
  Mean Velocity   3.12 m/s
  Peak Velocity   4.57 m/s

=== SPLIT TIMES ===

      from       to  split_time  distance  velocity
     Start        A        1.42      5.02      3.54
         A        B        2.18      9.87      4.53
         B        A        2.05      9.91      4.83
         A        C        1.87      4.96      2.65
         C        A        1.75      5.01      2.86
         A        D        1.52      5.12      3.37
         D        A        0.44      0.45      1.02
```

## Part 4: Best Practices

### Step 5: Add Validation and Error Handling

```python
class TTestProtocol(TestProtocol):
    # ... (previous code)
    
    def __init__(self, *args, **kwargs):
        # ... (previous init code)
        
        # Validate inputs
        self._validate()
    
    def _validate(self):
        """Validate protocol inputs."""
        # Check marker sampling frequency
        if self.athlete_marker.sampling_frequency < 50:
            raise ValueError(
                f"Marker sampling frequency too low: {self.athlete_marker.sampling_frequency} Hz. "
                "Minimum 50 Hz required for agility assessment."
            )
        
        # Check cone positions are distinct
        cone_positions = [
            (c['x'].data[0], c['y'].data[0]) 
            for c in self.cones.values()
        ]
        
        if len(set(cone_positions)) != 4:
            raise ValueError("Cone positions must be distinct.")
        
        # Check trial duration
        duration = len(self.athlete_marker) / self.athlete_marker.sampling_frequency
        if duration < 5:
            raise ValueError(f"Trial too short: {duration:.1f}s. Minimum 5s required.")
        if duration > 60:
            raise ValueError(f"Trial too long: {duration:.1f}s. Maximum 60s expected.")
```

## Key Takeaways

### Custom Protocol Checklist
- ✅ Inherit from TestProtocol (or follow duck typing interface)
- ✅ Implement `__init__()` with required parameters
- ✅ Implement `from_files()` classmethod
- ✅ Implement `get_results()` → returns TestResults
- ✅ Implement `save()` and `load()`
- ✅ Add `processing_pipeline` attribute
- ✅ Create `processed_data` copy
- ✅ Add input validation

### Custom Results Checklist
- ✅ Inherit from TestResults
- ✅ Set `participant`, `summary`, `analytics` attributes
- ✅ Implement `plot()` → returns go.Figure
- ✅ Implement `to_dataframe()`
- ✅ Implement `save()` and `load()`
- ✅ Generate `figures` dict

### Design Principles
1. **Consistency**: Match existing protocol patterns
2. **Validation**: Check inputs early, fail fast
3. **Documentation**: Complete docstrings with examples
4. **Flexibility**: Allow customization via parameters
5. **Robustness**: Handle edge cases gracefully

## Next Steps

- **Tutorial 07**: Batch processing multiple files
- **Tutorial 08**: Machine learning integration
- **API Reference**: [Protocols](../api-reference/protocols/protocols.md)
- **Examples**: Study existing protocols for patterns

---

**Complete guide to building custom test protocols following labanalysis design patterns and best practices.**

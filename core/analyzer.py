from core.ports import DataLoaderPort, ResultDisplayPort

class OcelAnalyzer:
    def __init__(self, loader: DataLoaderPort, display: ResultDisplayPort):
        self.loader = loader
        self.display = display

    def generate_overview(self, source_path: str):
        # 1. Use the injected adapter to get the data
        raw_data = self.loader.load(source_path)
        
        # The analyzer operates on the normalized representation, regardless
        # of whether the source was SQLite or JSON.
        stats = {
            "Report Type": "Database Overview",
            "Source": source_path,
            "Total Events": len(raw_data.events),
            "Total Objects": len(raw_data.objects),
            "Event-Object Links": len(raw_data.event_object_links),
            "Object-Object Links": len(raw_data.object_object_links),
        }
        
        # 3. Use the injected display adapter to show results
        self.display.render(stats)
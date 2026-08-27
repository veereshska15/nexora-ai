from schemas.telemetry import TelemetrySnapshot

class TelemetryService:
    @staticmethod
    async def get_system_telemetry() -> TelemetrySnapshot:
        return TelemetrySnapshot(
            cpu_percent=42.5,
            memory_percent=61.2,
            gpu_percent=73.8,
            latency_ms=41,
            active_connections=1,
            active_rag_documents=428,
            active_mcp_tools=17,
            security_threats=0,
            is_mock_data=True
        )

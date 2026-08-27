package com.nexora.controller;

import com.nexora.dto.response.ApiResponse;
import com.nexora.entity.AuditLog;
import com.nexora.entity.SecurityEvent;
import com.nexora.service.AuditService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/audit")
@RequiredArgsConstructor
public class AuditController {

    private final AuditService auditService;

    @GetMapping("/logs")
    public ResponseEntity<ApiResponse<List<AuditLog>>> getAuditLogs(HttpServletRequest servletRequest) {
        String requestId = servletRequest.getHeader("X-Request-ID");
        List<AuditLog> logs = auditService.getRecentAuditLogs();
        return ResponseEntity.ok(ApiResponse.success(logs, "Recent audit logs fetched", requestId != null ? requestId : "audit"));
    }

    @GetMapping("/security-events")
    public ResponseEntity<ApiResponse<List<SecurityEvent>>> getSecurityEvents(HttpServletRequest servletRequest) {
        String requestId = servletRequest.getHeader("X-Request-ID");
        List<SecurityEvent> events = auditService.getRecentSecurityEvents();
        return ResponseEntity.ok(ApiResponse.success(events, "Recent security events fetched", requestId != null ? requestId : "sec-events"));
    }
}

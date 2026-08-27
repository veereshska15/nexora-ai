package com.nexora.service;

import com.nexora.entity.AuditLog;
import com.nexora.entity.SecurityEvent;
import com.nexora.repository.AuditLogRepository;
import com.nexora.repository.SecurityEventRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class AuditService {

    private final AuditLogRepository auditLogRepository;
    private final SecurityEventRepository securityEventRepository;

    @Transactional
    public void logAuditEvent(UUID userId, String eventType, String description, String ipAddress, String requestId) {
        AuditLog auditLog = AuditLog.builder()
                .userId(userId)
                .eventType(eventType)
                .description(description)
                .ipAddress(ipAddress)
                .requestId(requestId)
                .build();

        auditLogRepository.save(auditLog);
    }

    @Transactional
    public void logSecurityEvent(UUID userId, String eventType, String severity, String details, String ipAddress, String requestId) {
        SecurityEvent securityEvent = SecurityEvent.builder()
                .userId(userId)
                .eventType(eventType)
                .severity(severity)
                .details(details)
                .ipAddress(ipAddress)
                .requestId(requestId)
                .build();

        securityEventRepository.save(securityEvent);
    }

    @Transactional(readOnly = true)
    public List<AuditLog> getRecentAuditLogs() {
        return auditLogRepository.findTop100ByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public List<SecurityEvent> getRecentSecurityEvents() {
        return securityEventRepository.findTop100ByOrderByCreatedAtDesc();
    }
}

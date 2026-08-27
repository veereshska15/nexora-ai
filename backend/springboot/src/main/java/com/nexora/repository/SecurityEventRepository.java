package com.nexora.repository;

import com.nexora.entity.SecurityEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface SecurityEventRepository extends JpaRepository<SecurityEvent, UUID> {
    List<SecurityEvent> findBySeverityOrderByCreatedAtDesc(String severity);
    List<SecurityEvent> findTop100ByOrderByCreatedAtDesc();
}

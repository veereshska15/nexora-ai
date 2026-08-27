package com.nexora.repository;

import com.nexora.entity.UserSession;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface SessionRepository extends JpaRepository<UserSession, UUID> {
    Optional<UserSession> findByRefreshTokenHashAndIsRevokedFalse(String refreshTokenHash);
    List<UserSession> findByUserIdAndIsRevokedFalse(UUID userId);
}

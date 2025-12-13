// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17; // Updated pragma

/// @title AlertRegistry
/// @notice Store a unique bytes32 hash for each alert for tamper-proofing
contract AlertRegistry {
    address public owner;

    struct Alert {
        bytes32 alertHash;
        uint256 timestamp;
        string asset_id;
        bool synthetic; // <-- ADDED: The new boolean flag
        address submitter;
    }

    // Storage remains the original mapping
    mapping(bytes32 => Alert) public alerts;

    // Event updated to include the synthetic flag
    event AlertRegistered(
        address indexed submitter,
        bytes32 indexed alertHash,
        uint256 timestamp,
        string asset_id,
        bool synthetic // <-- ADDED: Event parameter
    );

    constructor() {
        owner = msg.sender;
    }

    /// @notice Register a new alert by bytes32 hash, asset id, and synthetic flag
    /// @dev Will revert if the same alertHash was already registered
    // Function signature updated to accept synthetic and a timestamp (to save it directly)
    function registerAlert(
        bytes32 alertHash,
        uint256 timestamp, // <-- ADDED: Allow client to pass the timestamp
        string calldata asset_id,
        bool synthetic // <-- ADDED: New parameter
    ) external returns (bool) {
        require(alerts[alertHash].timestamp == 0, "Alert already registered");

        alerts[alertHash] = Alert(
            alertHash,
            timestamp, // Use the provided timestamp
            asset_id,
            synthetic, // Store the new flag
            msg.sender
        );

        emit AlertRegistered(
            msg.sender,
            alertHash,
            timestamp,
            asset_id,
            synthetic // Emit the new flag
        );
        return true;
    }

    /// @notice Read alert metadata by alertHash
    // Return signature updated to include the synthetic flag
    function getAlert(bytes32 alertHash) external view returns (bytes32, uint256, string memory, bool, address) {
        Alert memory a = alerts[alertHash];
        // Ensure the synthetic flag is returned
        return (a.alertHash, a.timestamp, a.asset_id, a.synthetic, a.submitter);
    }
}
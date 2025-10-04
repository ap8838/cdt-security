// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title AlertRegistry
/// @notice Store a unique bytes32 hash for each alert for tamper-proofing
contract AlertRegistry {
    address public owner;

    struct Alert {
        bytes32 alertHash;
        uint256 timestamp;
        string asset_id;
        address submitter;
    }

    mapping(bytes32 => Alert) public alerts;

    event AlertRegistered(address indexed submitter, bytes32 indexed alertHash, uint256 timestamp, string asset_id);

    constructor() {
        owner = msg.sender;
    }

    /// @notice Register a new alert by bytes32 hash and asset id
    /// @dev Will revert if the same alertHash was already registered
    function registerAlert(bytes32 alertHash, string calldata asset_id) external returns (bool) {
        require(alerts[alertHash].timestamp == 0, "Alert already registered");
        alerts[alertHash] = Alert(alertHash, block.timestamp, asset_id, msg.sender);
        emit AlertRegistered(msg.sender, alertHash, block.timestamp, asset_id);
        return true;
    }

    /// @notice Read alert metadata by alertHash
    function getAlert(bytes32 alertHash) external view returns (bytes32, uint256, string memory, address) {
        Alert memory a = alerts[alertHash];
        return (a.alertHash, a.timestamp, a.asset_id, a.submitter);
    }
}

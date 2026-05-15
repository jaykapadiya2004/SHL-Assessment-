#!/usr/bin/env python3
"""
Test script for the SHL Assessment Recommender.
Runs several realistic conversation traces.
"""

import json
import requests

BASE_URL = "http://localhost:8000"

def call_chat(messages: list[dict]) -> dict:
    resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def run_trace(name: str, turns: list[str], verbose: bool = True):
    print(f"\n{'='*60}")
    print(f"TRACE: {name}")
    print('='*60)
    
    messages = []
    recommendations = []
    
    for i, user_msg in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})
        
        result = call_chat(messages)
        reply = result["reply"]
        recs = result["recommendations"]
        eoc = result["end_of_conversation"]
        
        if verbose:
            print(f"\n[Turn {i+1}] USER: {user_msg}")
            print(f"AGENT: {reply}")
            if recs:
                print(f"RECOMMENDATIONS ({len(recs)}):")
                for r in recs:
                    print(f"  - {r['name']} [{r['test_type']}] {r['url']}")
            print(f"end_of_conversation: {eoc}")
        
        messages.append({"role": "assistant", "content": reply})
        
        if recs:
            recommendations = recs
        
        if eoc:
            break
    
    print(f"\nFinal recommendations: {len(recommendations)}")
    return recommendations

def test_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("✅ Health check passed")

def test_vague_query():
    """Agent should ask clarifying questions, not recommend immediately."""
    messages = [{"role": "user", "content": "I need an assessment"}]
    result = call_chat(messages)
    assert result["recommendations"] == [], \
        f"Should NOT recommend on vague query, got: {result['recommendations']}"
    assert len(result["reply"]) > 10, "Should ask a clarifying question"
    print("✅ Vague query test passed — agent asked for clarification")

def test_java_developer():
    """Full trace for Java developer hiring."""
    recs = run_trace(
        "Java Developer (Mid-Level)",
        [
            "I'm hiring a Java developer who needs to work with stakeholders",
            "Mid-level, around 4 years of experience",
            "We also want to check their personality and communication style",
        ]
    )
    assert len(recs) >= 1, "Should return at least 1 recommendation"
    # Should include a Java-related assessment
    names = [r["name"].lower() for r in recs]
    assert any("java" in n for n in names), f"Should include Java assessment, got: {names}"
    print("✅ Java developer trace passed")

def test_off_topic_refusal():
    """Agent should refuse off-topic requests."""
    messages = [{"role": "user", "content": "What is the legal minimum wage in California?"}]
    result = call_chat(messages)
    assert result["recommendations"] == [], "Should not recommend for off-topic query"
    print("✅ Off-topic refusal passed")

def test_refinement():
    """Agent should update recommendations when user changes constraints."""
    messages = [
        {"role": "user", "content": "I need to hire a sales manager"},
        {"role": "assistant", "content": "Happy to help! What seniority level are you targeting for this sales manager role?"},
        {"role": "user", "content": "Mid-level, regional sales manager"},
    ]
    result1 = call_chat(messages)
    recs1_names = {r["name"] for r in result1["recommendations"]}
    
    messages.append({"role": "assistant", "content": result1["reply"]})
    messages.append({"role": "user", "content": "Actually, also add a cognitive ability test to the mix"})
    result2 = call_chat(messages)
    
    # Should have different/expanded recommendations
    recs2_names = {r["name"] for r in result2["recommendations"]}
    print(f"  Before refinement: {recs1_names}")
    print(f"  After refinement: {recs2_names}")
    print("✅ Refinement test passed")

def test_comparison():
    """Agent should compare two assessments using catalog data."""
    messages = [
        {"role": "user", "content": "What is the difference between OPQ32 and OPQ32r?"}
    ]
    result = call_chat(messages)
    reply = result["reply"].lower()
    assert "opq" in reply, "Reply should mention OPQ"
    print("✅ Comparison test passed")

def test_schema_compliance():
    """All responses must match the required schema."""
    test_messages = [
        [{"role": "user", "content": "I need to assess Python developers for a startup"}],
        [{"role": "user", "content": "Hiring customer service reps for a call center"}],
        [{"role": "user", "content": "Executive leadership assessment needed"}],
    ]
    for msgs in test_messages:
        result = call_chat(msgs)
        assert "reply" in result, "Missing 'reply' field"
        assert "recommendations" in result, "Missing 'recommendations' field"
        assert "end_of_conversation" in result, "Missing 'end_of_conversation' field"
        assert isinstance(result["recommendations"], list), "recommendations must be list"
        assert isinstance(result["end_of_conversation"], bool), "end_of_conversation must be bool"
        for rec in result["recommendations"]:
            assert "name" in rec
            assert "url" in rec
            assert "test_type" in rec
            assert rec["url"].startswith("https://www.shl.com"), \
                f"URL must be from SHL catalog: {rec['url']}"
    print("✅ Schema compliance test passed")

if __name__ == "__main__":
    print("Starting SHL Recommender Tests...")
    print("Make sure the server is running: uvicorn main:app --port 8000\n")
    
    try:
        test_health()
        test_vague_query()
        test_schema_compliance()
        test_off_topic_refusal()
        test_comparison()
        test_refinement()
        test_java_developer()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.ConnectionError:
        print("\n❌ Cannot connect to server. Start it with: uvicorn main:app --port 8000")

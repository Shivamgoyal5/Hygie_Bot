
"use client";

import { useState, useEffect } from "react";
import PromptCard from "./PromptCard";

const PromptCardList = ({ data, handleTagClick }) => {
  return (
    <div className='mt-16 prompt_layout'>
      {data.map((post) => (
        <PromptCard
          key={post._id}
          post={post}
          handleTagClick={handleTagClick}
        />
      ))}
    </div>
  );
};

const Feed = () => {
  const [allPosts, setAllPosts] = useState([]);
  const [searchText, setSearchText] = useState("");
  const [searchTimeout, setSearchTimeout] = useState(null);
  const [searchedResults, setSearchedResults] = useState([]);
  const [helloWorldMessage, setHelloWorldMessage] = useState("");

  const fetchPosts = async () => {
    const response = await fetch("/api/prompt");
    const data = await response.json();

    setAllPosts(data);
  };

  useEffect(() => {
    fetchPosts();
  }, []);

  const filterPrompts = (searchtext) => {
    const regex = new RegExp(searchtext, "i");
    return allPosts.filter(
      (item) =>
        regex.test(item.creator.username) ||
        regex.test(item.tag) ||
        regex.test(item.prompt)
    );
  };

  const handleSearchChange = (e) => {
    clearTimeout(searchTimeout);
    setSearchText(e.target.value);

    setSearchTimeout(
      setTimeout(() => {
        const searchResult = filterPrompts(e.target.value);
        setSearchedResults(searchResult);
      }, 500)
    );
  };

  const handleTagClick = (tagName) => {
    setSearchText(tagName);

    const searchResult = filterPrompts(tagName);
    setSearchedResults(searchResult);
  };

  const handleEnterPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleFormSubmit();
    }
  };
  
  const handleFormSubmit = async () => {
    try {
      // Make the API call
      const apiUrl = `http://127.0.0.1:8000/generate_response/?prompt=${encodeURIComponent(searchText)}`;
      const response = await fetch(apiUrl);
      
      // Check if the response was successful
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      // Parse the JSON response
      const responseData = await response.json();

      // Extract the 'response' field from the API response
      const apiResponse = responseData.response;

      // Set the response in the component's state
      setHelloWorldMessage(apiResponse);
    } catch (error) {
      console.error('Error fetching data:', error.message);
      // Handle errors if needed
    }
  };


  return (
    <section className='feed'>
      <form className='relative w-full flex-center' onSubmit={handleFormSubmit}>
        <input
          type='text'
          placeholder='Search for a tag or a username'
          value={searchText}
          onChange={handleSearchChange}
          onKeyPress={handleEnterPress}
          required
          className='search_input peer'
        />
        <button type="submit" style={{ display: 'none' }}></button>
      </form>

      {/* Display OpenAI response instead of Hello, world! message */}
      {helloWorldMessage && <div>{helloWorldMessage}</div>}

      {/* All Prompts */}
      {searchText ? (
        <PromptCardList
          data={searchedResults}
          handleTagClick={handleTagClick}
        />
      ) : (
        <PromptCardList data={allPosts} handleTagClick={handleTagClick} />
      )}
    </section>
  );
};

export default Feed;
import React from 'react'
import { NavLink } from 'react-router-dom'

const Tool = ({tool}) => {

  return (
    <NavLink to={tool.link} className="border border-gray-300 rounded-lg overflow-hidden hover:shadow-lg shadow-special-red/10 transition-shadow duration-300">
      <div className="flex flex-col items-center">
        <img src={tool.image} alt={tool.title} className="w-32 h-32 object-contain rounded m-4" />
        <div className='bg-special-red/15 p-4 '>
          <h2 className="text-special-red md:text-lg line-clamp-2 md:line-clamp-1 mb-2" title={tool.title}>{tool.title}</h2>
          <p className="text-gray-600 mb-2 text-xs md:text-sm line-clamp-2" title={tool.description}>{tool.description}</p>
        </div>
      </div>
    </NavLink>
  )
}

export default Tool